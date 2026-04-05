from database import get_connection

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        baseline_model TEXT,
        updated_model TEXT,
        flip_rate REAL,
        confidence_shift REAL,
        feature_drift REAL,
        subgroup_risk REAL,
        bias_severity REAL,
        final_risk REAL,
        decision TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()
    print("Database initialized successfully.")