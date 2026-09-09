-- ====================================================================
-- OPERATIONAL FRAUD MONITORING & RISK TIERS SQL
-- Joins predictions, alerts, and transaction records
-- ====================================================================

-- 1. High-Risk Alert Feed with Predictions & Transaction Details
SELECT 
    a.alert_id,
    a.transaction_id,
    t.amount,
    a.fraud_probability,
    a.risk_level,
    a.alert_status,
    p.threshold_used,
    p.model_version,
    a.timestamp AS alert_timestamp
FROM fraud_alerts a
JOIN transactions t ON a.transaction_id = t.transaction_id
JOIN fraud_predictions p ON a.transaction_id = p.transaction_id
WHERE a.risk_level IN ('HIGH', 'CRITICAL')
ORDER BY a.fraud_probability DESC, a.timestamp DESC;

-- 2. Risk Tier Breakdown Summary
SELECT 
    risk_level,
    COUNT(*) AS alert_count,
    ROUND(AVG(amount), 2) AS avg_alert_amount,
    ROUND(SUM(amount), 2) AS total_alert_exposure,
    ROUND(AVG(fraud_probability), 4) AS avg_model_confidence
FROM fraud_alerts
GROUP BY risk_level
ORDER BY 
    CASE risk_level
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        WHEN 'MEDIUM' THEN 3
        WHEN 'LOW' THEN 4
        ELSE 5
    END;

-- 3. Model Accuracy & Confusion Matrix from Predictions
SELECT 
    t.class AS actual_label,
    p.prediction AS predicted_label,
    COUNT(*) AS count,
    CASE 
        WHEN t.class = 1 AND p.prediction = 1 THEN 'True Positive (Fraud Caught)'
        WHEN t.class = 0 AND p.prediction = 0 THEN 'True Negative (Legit Allowed)'
        WHEN t.class = 0 AND p.prediction = 1 THEN 'False Positive (False Alarm)'
        WHEN t.class = 1 AND p.prediction = 0 THEN 'False Negative (Fraud Missed)'
    END AS classification_category
FROM transactions t
JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
GROUP BY t.class, p.prediction;

-- 4. Hourly Velocity of Critical Alerts (Window Running Total)
WITH critical_alerts AS (
    SELECT 
        CAST(strftime('%H', timestamp) AS INTEGER) AS alert_hour,
        amount
    FROM fraud_alerts
    WHERE risk_level = 'CRITICAL'
)
SELECT 
    alert_hour,
    COUNT(*) AS critical_count,
    ROUND(SUM(amount), 2) AS hourly_loss_prevented,
    SUM(COUNT(*)) OVER (ORDER BY alert_hour) AS cumulative_critical_alerts
FROM critical_alerts
GROUP BY alert_hour
ORDER BY alert_hour;
