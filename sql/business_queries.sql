-- ====================================================================
-- BUSINESS ANALYTICS SQL QUERIES: CREDIT CARD FRAUD DETECTION
-- Advanced SQL demonstrating CTEs, Window Functions, CASE, Aggregates
-- ====================================================================

-- 1. Total Transactions Count
SELECT 
    COUNT(*) AS total_transactions
FROM transactions;

-- 2. Total Fraud Transactions Count
SELECT 
    COUNT(*) AS total_fraud_transactions
FROM transactions
WHERE class = 1;

-- 3. Fraud Percentage Rate
SELECT 
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_percentage
FROM transactions;

-- 4. Total Transaction Amount
SELECT 
    ROUND(SUM(amount), 2) AS total_transaction_amount
FROM transactions;

-- 5. Fraud Transaction Amount
SELECT 
    ROUND(SUM(amount), 2) AS total_fraud_amount,
    ROUND(AVG(amount), 2) AS avg_fraud_amount,
    ROUND(MAX(amount), 2) AS max_fraud_amount
FROM transactions
WHERE class = 1;

-- 6. Average Transaction Amount (Legit vs Fraud Breakdown)
SELECT 
    CASE 
        WHEN class = 1 THEN 'Fraudulent' 
        ELSE 'Legitimate' 
    END AS transaction_type,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount,
    ROUND(SUM(amount), 2) AS total_amount
FROM transactions
GROUP BY class;

-- 7. Fraud Rate by Amount Bracket (Cohort Analysis with CASE & GROUP BY)
WITH amount_brackets AS (
    SELECT 
        transaction_id,
        amount,
        class,
        CASE 
            WHEN amount < 20.0 THEN '1. Micro (< $20)'
            WHEN amount >= 20.0 AND amount < 100.0 THEN '2. Small ($20 - $100)'
            WHEN amount >= 100.0 AND amount < 500.0 THEN '3. Medium ($100 - $500)'
            WHEN amount >= 500.0 AND amount < 2000.0 THEN '4. Large ($500 - $2,000)'
            ELSE '5. Jumbo ($2,000+)'
        END AS amount_tier
    FROM transactions
)
SELECT 
    amount_tier,
    COUNT(*) AS total_tx_in_tier,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_tx_in_tier,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS tier_fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_dollar_volume,
    ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 2) AS fraud_dollar_loss
FROM amount_brackets
GROUP BY amount_tier
ORDER BY amount_tier;

-- 8. Fraud Rate Over Time (Hourly Trend using Floor Math)
WITH hourly_tx AS (
    SELECT 
        CAST((time / 3600) AS INTEGER) % 24 AS hour_of_day,
        class,
        amount
    FROM transactions
)
SELECT 
    hour_of_day,
    COUNT(*) AS tx_count,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS hourly_fraud_rate_pct,
    ROUND(SUM(amount), 2) AS hourly_total_volume
FROM hourly_tx
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 9. Top 15 Highest-Value Fraudulent Transactions
SELECT 
    transaction_id,
    ROUND(time / 3600.0, 2) AS elapsed_hours,
    amount,
    class
FROM transactions
WHERE class = 1
ORDER BY amount DESC
LIMIT 15;

-- 10. Daily Transaction Volume
SELECT 
    CAST((time / 86400) + 1 AS INTEGER) AS simulation_day,
    COUNT(*) AS daily_total_transactions,
    ROUND(SUM(amount), 2) AS daily_total_amount,
    ROUND(AVG(amount), 2) AS daily_avg_ticket_size
FROM transactions
GROUP BY CAST((time / 86400) + 1 AS INTEGER)
ORDER BY simulation_day;

-- 11. Daily Fraud Count
SELECT 
    CAST((time / 86400) + 1 AS INTEGER) AS simulation_day,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS daily_fraud_count,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS daily_fraud_rate_pct
FROM transactions
GROUP BY CAST((time / 86400) + 1 AS INTEGER)
ORDER BY simulation_day;

-- 12. Daily Fraud Dollar Amount & Exposure
SELECT 
    CAST((time / 86400) + 1 AS INTEGER) AS simulation_day,
    ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 2) AS daily_fraud_amount,
    ROUND(AVG(CASE WHEN class = 1 THEN amount ELSE NULL END), 2) AS daily_avg_fraud_ticket
FROM transactions
GROUP BY CAST((time / 86400) + 1 AS INTEGER)
ORDER BY simulation_day;

-- 13. Hourly Fraud Patterns (Identifying Peak Risk Hours with HAVING)
WITH hourly_analysis AS (
    SELECT 
        CAST((time / 3600) AS INTEGER) % 24 AS hour_of_day,
        COUNT(*) AS total_tx,
        SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_tx
    FROM transactions
    GROUP BY CAST((time / 3600) AS INTEGER) % 24
)
SELECT 
    hour_of_day,
    total_tx,
    fraud_tx,
    ROUND(100.0 * fraud_tx / total_tx, 4) AS fraud_rate_pct
FROM hourly_analysis
WHERE fraud_tx > 0
ORDER BY fraud_rate_pct DESC;

-- 14. Top Suspicious Transaction Periods (Window Functions: RANK and Rolling Fraud Count)
WITH period_summary AS (
    SELECT 
        CAST((time / 7200) AS INTEGER) AS two_hour_block,
        COUNT(*) AS tx_count,
        SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_count,
        ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 2) AS fraud_loss
    FROM transactions
    GROUP BY CAST((time / 7200) AS INTEGER)
)
SELECT 
    two_hour_block,
    two_hour_block * 2 AS block_start_hour,
    (two_hour_block * 2) + 2 AS block_end_hour,
    tx_count,
    fraud_count,
    fraud_loss,
    ROUND(100.0 * fraud_count / tx_count, 4) AS fraud_rate_pct,
    RANK() OVER (ORDER BY fraud_count DESC) AS fraud_density_rank,
    SUM(fraud_count) OVER (ORDER BY two_hour_block) AS cumulative_fraud_count
FROM period_summary
ORDER BY fraud_density_rank ASC
LIMIT 10;
