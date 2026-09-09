-- Real-Time Credit Card Fraud Detection Database Schema
-- Compatible with SQLite, PostgreSQL, and MySQL

-- 1. Raw / Ingested Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    time FLOAT NOT NULL,
    v1 FLOAT NOT NULL,
    v2 FLOAT NOT NULL,
    v3 FLOAT NOT NULL,
    v4 FLOAT NOT NULL,
    v5 FLOAT NOT NULL,
    v6 FLOAT NOT NULL,
    v7 FLOAT NOT NULL,
    v8 FLOAT NOT NULL,
    v9 FLOAT NOT NULL,
    v10 FLOAT NOT NULL,
    v11 FLOAT NOT NULL,
    v12 FLOAT NOT NULL,
    v13 FLOAT NOT NULL,
    v14 FLOAT NOT NULL,
    v15 FLOAT NOT NULL,
    v16 FLOAT NOT NULL,
    v17 FLOAT NOT NULL,
    v18 FLOAT NOT NULL,
    v19 FLOAT NOT NULL,
    v20 FLOAT NOT NULL,
    v21 FLOAT NOT NULL,
    v22 FLOAT NOT NULL,
    v23 FLOAT NOT NULL,
    v24 FLOAT NOT NULL,
    v25 FLOAT NOT NULL,
    v26 FLOAT NOT NULL,
    v27 FLOAT NOT NULL,
    v28 FLOAT NOT NULL,
    amount FLOAT NOT NULL,
    class INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Real-Time Fraud Predictions Table
CREATE TABLE IF NOT EXISTS fraud_predictions (
    prediction_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    fraud_probability FLOAT NOT NULL,
    prediction INTEGER NOT NULL,
    threshold_used FLOAT NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
);

-- 3. Fraud Alerts Table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    amount FLOAT NOT NULL,
    fraud_probability FLOAT NOT NULL,
    risk_level VARCHAR(16) NOT NULL, -- LOW, MEDIUM, HIGH, CRITICAL
    alert_status VARCHAR(16) DEFAULT 'NEW', -- NEW, REVIEWED, RESOLVED, DISMISSED
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
);

-- 4. Model Metrics Table
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id VARCHAR(64) PRIMARY KEY,
    model_name VARCHAR(64) NOT NULL,
    accuracy FLOAT NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    roc_auc FLOAT NOT NULL,
    pr_auc FLOAT NOT NULL,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for high-throughput lookup and real-time dashboarding
CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(time);
CREATE INDEX IF NOT EXISTS idx_transactions_class ON transactions(class);
CREATE INDEX IF NOT EXISTS idx_predictions_tx ON fraud_predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_alerts_risk ON fraud_alerts(risk_level);
CREATE INDEX IF NOT EXISTS idx_alerts_time ON fraud_alerts(timestamp);
