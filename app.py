import os

import sqlite3
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

DB_FILE = "fraud_analytics.db"
CSV_FILE = "creditcard.csv"

# Global references for cached data and trained model
MODEL = None
FEATURE_NAMES = []
EVALUATION_METRICS = {}

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            amount REAL,
            fraud_probability REAL,
            prediction TEXT,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_data_and_train():
    global MODEL, FEATURE_NAMES, EVALUATION_METRICS
    if not os.path.exists(CSV_FILE):
        print(f"Warning: {CSV_FILE} not found. Running with mock evaluation metrics.")
        return

    print("Loading creditcard.csv and training baseline model...")
    df = pd.read_csv(CSV_FILE)
    
    # Simple sampling for fast startup if dataset is large
    if len(df) > 50000:
        fraud_df = df[df['Class'] == 1]
        legit_df = df[df['Class'] == 0].sample(n=45000, random_state=42)
        df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42)

    X = df.drop(columns=['Class'])
    y = df['Class']
    FEATURE_NAMES = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    MODEL = clf

    # Calculate real evaluation metrics
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    # Downsample curves for efficient JSON transmission
    step = max(1, len(fpr) // 50)
    roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::step], tpr[::step])]
    pr_data = [{"precision": round(float(p), 4), "recall": round(float(r), 4)} for p, r in zip(precision[::step], recall[::step])]

    EVALUATION_METRICS = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": {
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),
            "fn": int(cm[1][0]), "tp": int(cm[1][1])
        },
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "total_test_samples": int(len(y_test)),
        "total_fraud_samples": int(y_test.sum())
    }
    print("Model training complete and metrics computed.")

# Routes
@app.route('/')
def serve_index():
    return send_from_directory('.', 'model_evaluation_fraud_detection_dashboard.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(EVALUATION_METRICS)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    amount = float(data.get('amount', 0))
    time_val = float(data.get('time', 0))

    if MODEL is not None:
        # Build array with all feature columns initialized to zero, updating available fields
        input_vec = np.zeros((1, len(FEATURE_NAMES)))
        if 'Amount' in FEATURE_NAMES:
            input_vec[0, FEATURE_NAMES.index('Amount')] = amount
        if 'Time' in FEATURE_NAMES:
            input_vec[0, FEATURE_NAMES.index('Time')] = time_val
        
        # Fill in extra V-features if sent by dashboard
        for i in range(1, 29):
            col = f'V{i}'
            if col in FEATURE_NAMES and col in data:
                input_vec[0, FEATURE_NAMES.index(col)] = float(data[col])

        pred_class = int(MODEL.predict(input_vec)[0])
        prob = float(MODEL.predict_proba(input_vec)[0][1])
    else:
        # Fallback heuristic if dataset hasn't finished loading
        prob = 0.85 if amount > 2000 else 0.05
        pred_class = 1 if prob > 0.5 else 0

    fraud_prob_pct = round(prob * 100, 2)
    status_label = "Fraud" if pred_class == 1 else "Legitimate"

    # Save to SQLite DB
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_history (amount, fraud_probability, prediction, status)
        VALUES (?, ?, ?, ?)
    """, (amount, fraud_prob_pct, status_label, "FLAGGED" if pred_class == 1 else "APPROVED"))
    conn.commit()
    conn.close()

    return jsonify({
        "is_fraud": bool(pred_class == 1),
        "fraud_probability": fraud_prob_pct,
        "status": status_label
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, amount, fraud_probability, prediction, status FROM prediction_history ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    history = [
        {
            "id": r[0],
            "timestamp": r[1],
            "amount": r[2],
            "fraud_probability": r[3],
            "prediction": r[4],
            "status": r[5]
        }
        for r in rows
    ]
    return jsonify(history)

if __name__ == '__main__':
    init_db()
    load_data_and_train()
    app.run(host='127.0.0.1', port=5000, debug=True)