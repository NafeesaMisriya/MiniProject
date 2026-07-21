import os
import tempfile
import joblib
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analysis.compute_metrics import compute_metrics
from analysis.subgroup_risk import compute_subgroup_risk
from analysis.bias_severity import compute_bias_severity
from backend.database import get_connection

app = FastAPI(title="ModelGuard AI Risk Engine API")

# Auto-initialize SQLite database tables on startup
try:
    from backend.init_db import create_tables
    create_tables()
    print("Database tables initialized successfully.")
except Exception as _e:
    print(f"Database initialization skipped/failed: {_e}")


# Enable CORS for local development with Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_MODEL_PATH = "models/current.pkl"

def read_current_model():
    try:
        with open(CURRENT_MODEL_PATH, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "None set"

def write_current_model(model_name: str):
    os.makedirs("models", exist_ok=True)
    with open(CURRENT_MODEL_PATH, "w") as f:
        f.write(f"Active Model: {model_name}")

def reset_current_model():
    os.makedirs("models", exist_ok=True)
    with open(CURRENT_MODEL_PATH, "w") as f:
        f.write("Active Model: None")


class ReviewActionRequest(BaseModel):
    action: str  # "accept" or "rollback"
    candidate_name: str
    baseline_name: str


@app.get("/api/active-model")
def get_active_model():
    return {"active_model": read_current_model()}


@app.post("/api/reset-model")
def post_reset_model():
    reset_current_model()
    return {"active_model": read_current_model()}


@app.post("/api/upload-baseline")
async def upload_baseline(
    file: Optional[UploadFile] = File(None),
    baseline_model_name: Optional[str] = Form(None)
):
    if not file and not baseline_model_name:
        raise HTTPException(status_code=400, detail="Please upload a model or select a pre-loaded model.")
    
    if file:
        if not file.filename.endswith(".pkl"):
            raise HTTPException(status_code=400, detail="Only .pkl files are allowed.")
        baseline_name = os.path.splitext(file.filename)[0]
    else:
        allowed_preloaded = ["model_v0", "model_v1", "model_v2", "model_v3"]
        if baseline_model_name not in allowed_preloaded:
            raise HTTPException(status_code=400, detail=f"Invalid pre-loaded model: {baseline_model_name}")
        baseline_name = baseline_model_name
        
    write_current_model(baseline_name)
    return {"active_model": read_current_model(), "baseline_name": baseline_name}


@app.post("/api/run-analysis")
async def run_analysis(
    baseline_file: Optional[UploadFile] = File(None),
    candidate_file: Optional[UploadFile] = File(None),
    baseline_model_name: Optional[str] = Form(None),
    candidate_model_name: Optional[str] = Form(None)
):
    if not baseline_file and not baseline_model_name:
        raise HTTPException(status_code=400, detail="Please upload a baseline model or select a pre-loaded model.")
    if not candidate_file and not candidate_model_name:
        raise HTTPException(status_code=400, detail="Please upload a candidate model or select a pre-loaded model.")

    tmp_baseline = None
    tmp_candidate = None

    try:
        # Load baseline model
        if baseline_file:
            if not baseline_file.filename.endswith(".pkl"):
                raise HTTPException(status_code=400, detail="Only .pkl files are allowed.")
            baseline_name = os.path.splitext(baseline_file.filename)[0]
            tmp_baseline = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            content_base = await baseline_file.read()
            tmp_baseline.write(content_base)
            tmp_baseline.close()
            baseline_model = joblib.load(tmp_baseline.name)
        else:
            allowed_preloaded = ["model_v0", "model_v1", "model_v2", "model_v3"]
            if baseline_model_name not in allowed_preloaded:
                raise HTTPException(status_code=400, detail=f"Invalid pre-loaded baseline model: {baseline_model_name}")
            baseline_name = baseline_model_name
            baseline_model = joblib.load(f"models/{baseline_model_name}.pkl")

        # Load candidate model
        if candidate_file:
            if not candidate_file.filename.endswith(".pkl"):
                raise HTTPException(status_code=400, detail="Only .pkl files are allowed.")
            candidate_name = os.path.splitext(candidate_file.filename)[0]
            tmp_candidate = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            content_cand = await candidate_file.read()
            tmp_candidate.write(content_cand)
            tmp_candidate.close()
            updated_model = joblib.load(tmp_candidate.name)
        else:
            allowed_preloaded = ["model_v0", "model_v1", "model_v2", "model_v3"]
            if candidate_model_name not in allowed_preloaded:
                raise HTTPException(status_code=400, detail=f"Invalid pre-loaded candidate model: {candidate_model_name}")
            candidate_name = candidate_model_name
            updated_model = joblib.load(f"models/{candidate_model_name}.pkl")

        # Load dataset
        X_test = joblib.load("data/X_test.pkl")
        y_test = joblib.load("data/y_test.pkl")

        # Compute metrics
        flip_rate, conf_shift, feature_drift = compute_metrics(
            baseline_model, updated_model, X_test
        )
        subgroup_risk = compute_subgroup_risk(
            baseline_model, updated_model, X_test, y_test, feature="mean radius"
        )
        bias_severity = compute_bias_severity(
            updated_model, X_test, y_test, feature="mean radius"
        )

        final_risk = (
            0.30 * flip_rate +
            0.25 * conf_shift +
            0.20 * feature_drift +
            0.15 * subgroup_risk +
            0.10 * bias_severity
        )

        decision = (
            "DEPLOY" if final_risk < 0.02 else
            "REVIEW" if final_risk < 0.07 else
            "ROLLBACK"
        )

        # Write active model based on decision
        if decision == "DEPLOY":
            write_current_model(candidate_name)
        elif decision == "ROLLBACK":
            write_current_model(baseline_name)

        # Log comparison to DB
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO comparisons
               (baseline_model, updated_model, flip_rate, confidence_shift,
                feature_drift, subgroup_risk, bias_severity, final_risk, decision)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                baseline_name, candidate_name,
                flip_rate, conf_shift,
                feature_drift, subgroup_risk,
                bias_severity, final_risk, decision
            )
        )
        conn.commit()
        conn.close()

        return {
            "flip_rate": flip_rate,
            "conf_shift": conf_shift,
            "feature_drift": feature_drift,
            "subgroup_risk": subgroup_risk,
            "bias_severity": bias_severity,
            "final_risk": final_risk,
            "decision": decision,
            "candidate_name": candidate_name,
            "baseline_name": baseline_name,
            "active_model": read_current_model()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        if tmp_baseline is not None:
            try:
                os.unlink(tmp_baseline.name)
            except Exception:
                pass
        if tmp_candidate is not None:
            try:
                os.unlink(tmp_candidate.name)
            except Exception:
                pass


@app.post("/api/review-action")
def review_action(req: ReviewActionRequest):
    if req.action == "accept":
        write_current_model(req.candidate_name)
    elif req.action == "rollback":
        write_current_model(req.baseline_name)
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'accept' or 'rollback'.")
    
    return {"active_model": read_current_model()}


@app.get("/api/history")
def get_history():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, baseline_model, updated_model, flip_rate, confidence_shift,
                      feature_drift, subgroup_risk, bias_severity, final_risk, decision, timestamp
               FROM comparisons
               ORDER BY id DESC"""
        )
        rows = cur.fetchall()
        
        history = []
        for r in rows:
            history.append({
                "id": r["id"],
                "baseline_model": r["baseline_model"],
                "updated_model": r["updated_model"],
                "flip_rate": r["flip_rate"],
                "confidence_shift": r["confidence_shift"],
                "feature_drift": r["feature_drift"],
                "subgroup_risk": r["subgroup_risk"],
                "bias_severity": r["bias_severity"],
                "final_risk": r["final_risk"],
                "decision": r["decision"],
                "timestamp": r["timestamp"]
            })
        conn.close()
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files of frontend if built (production mode)
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
