# ModelGuard AI Risk Engine

A premium model risk evaluation platform built with a **React frontend** and a **FastAPI backend**. It automatically measures and visualizes risk metrics (prediction flip rate, confidence shift, feature drift, subgroup risk, and bias severity) between a baseline model and a candidate model.

## Architecture

* **Frontend**: React SPA scaffolded with Vite and styled with a customized dark-theme glassmorphic design. Contains views for uploading models, analyzing risk metrics in interactive bar charts, and browsing run histories.
* **Backend**: FastAPI REST server handling model uploads, loading datasets, computing metrics using local analysis scripts, managing active models (`models/current.pkl`), and logging histories to a SQLite database.

## Prerequisites

Ensure you have Python 3.8+ and Node.js installed.

---

## Local Development & Setup

### 1. Install Dependencies

Install the python requirements:
```bash
pip install -r requirements.txt
```

### 2. Initialize the Database

Run the database setup script to initialize the SQLite database table:
```bash
python backend/init_db.py
```

### 3. Launch the Application

Double-click `run.bat` (on Windows) or run the following command in your terminal:
```bash
python -m uvicorn main:app --port 8000 --host 127.0.0.1
```
The application will start, and your web browser will automatically open `http://127.0.0.1:8000/`.

---

## Production Cloud Deployment

### 1. Backend on Render (Python FastAPI)
We use a Render Blueprint specification to deploy the backend automatically:
1. Go to [Render](https://render.com/) and create a new **Blueprint**.
2. Connect your Git repository. Render will automatically read the `render.yaml` configuration and set up the Python service.
3. Render will install dependencies and start uvicorn. Copy your backend service URL (e.g., `https://modelguard-backend.onrender.com`).

### 2. Frontend on Vercel (React Vite App)
1. Go to [Vercel](https://vercel.com/) and import your Git repository.
2. In Project Settings:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. Under **Environment Variables**, add:
   - **Key**: `VITE_API_URL`
   - **Value**: Your Render backend service URL.
4. Click **Deploy**. Vercel will compile and host the React SPA.

---

## API Documentation

* **`GET /api/active-model`**: Get the current active model name.
* **`POST /api/reset-model`**: Resets the active model name to `None`.
* **`POST /api/upload-baseline`**: Uploads and sets the active model to the baseline name.
* **`POST /api/run-analysis`**: Uploads baseline and candidate models, runs risk evaluation, saves results to the database, and promotes the model according to risk score.
* **`POST /api/review-action`**: Submits user action ("accept" or "rollback") for caution-range models.
* **`GET /api/history`**: Retrieves all previous runs from the SQLite database comparisons table.