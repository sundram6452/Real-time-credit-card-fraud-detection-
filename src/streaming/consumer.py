"""
Fraud Detection Real-Time Stream Consumer
Consumes transactions, executes real-time inference, flags risk tiers,
and persists predictions and business alerts to the database.
"""

import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import pandas as pd

from src.config import (
    USE_KAFKA,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    DEFAULT_FRAUD_THRESHOLD
)
from src.data.validator import validate_transaction_dict
from src.models.predict import predict_single_transaction
from src.streaming.kafka_simulator import SimulatedKafkaConsumer
from src.database.queries import (
    save_transaction,
    save_prediction,
    save_alert
)

def get_consumer(topic: str = KAFKA_TOPIC, use_kafka: bool = USE_KAFKA):
    """Factory returning either actual KafkaConsumer or SimulatedKafkaConsumer."""
    if use_kafka:
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset="latest",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=1000
            )
            print(f"[Consumer] Connected to production Kafka at {KAFKA_BOOTSTRAP_SERVERS} for topic '{topic}'")
            return consumer, "kafka"
        except Exception as e:
            print(f"[Consumer] Kafka connection failed ({e}). Falling back to simulated streaming layer.")

    consumer = SimulatedKafkaConsumer(
        topic,
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    print(f"[Consumer] Initialized Kafka-compatible simulated streaming consumer for topic '{topic}'.")
    return consumer, "simulator"

def format_fraud_alert_box(
    tx_id: str,
    amount: float,
    prob: float,
    risk_level: str,
    timestamp: str
) -> str:
    """Formats the standardized business fraud alert box."""
    return f"""
+-------------------------------------------------------------+
| [ALERT] FRAUD SUSPICION DETECTED                            |
+-------------------------------------------------------------+
| Transaction ID:    {tx_id:<40} |
| Amount:            ${amount:<39.2f} |
| Fraud Probability: {prob * 100.0:<38.1f}% |
| Risk Level:        {risk_level:<40} |
| Timestamp:         {timestamp:<40} |
+-------------------------------------------------------------+
"""

def start_consuming_transactions(
    topic: str = KAFKA_TOPIC,
    use_kafka: bool = USE_KAFKA,
    threshold: float = DEFAULT_FRAUD_THRESHOLD,
    max_records: Optional[int] = None,
    poll_timeout_ms: int = 1000,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> int:
    """
    Main consumption loop. Processes streaming transactions, detects fraud,
    issues formatted alerts, and persists records to the database.
    """
    consumer, mode = get_consumer(topic=topic, use_kafka=use_kafka)
    processed_count = 0
    alerts_count = 0

    print(f"[Consumer] Listening for transaction events (threshold={threshold}, mode={mode})...")

    try:
        while True:
            # Poll records
            if mode == "kafka":
                msg_dict = consumer.poll(timeout_ms=poll_timeout_ms)
                records = [rec.value for sublist in msg_dict.values() for rec in sublist]
            else:
                msg_dict = consumer.poll(timeout_ms=poll_timeout_ms, max_records=5)
                records = [rec.value for sublist in msg_dict.values() for rec in sublist]

            if not records:
                time.sleep(0.1)
                continue

            for payload in records:
                # 1. Validation
                is_valid, err_msg = validate_transaction_dict(payload, require_class=False)
                if not is_valid:
                    print(f"[Consumer] Rejected invalid transaction: {err_msg}")
                    continue

                # 2. Real-Time Inference
                pred_result = predict_single_transaction(payload, threshold=threshold)
                
                tx_id = pred_result["transaction_id"]
                prob = pred_result["fraud_probability"]
                is_fraud = pred_result["is_fraud"]
                risk_level = pred_result["risk_level"]
                amt = pred_result["amount"]
                ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                # 3. Persistence to Relational Database
                try:
                    save_transaction(payload)
                    save_prediction(pred_result)
                except Exception as e:
                    print(f"[Consumer] DB save error: {e}")

                # 4. Alert Generation
                if is_fraud or risk_level in ("HIGH", "CRITICAL"):
                    alerts_count += 1
                    alert_box = format_fraud_alert_box(tx_id, amt, prob, risk_level, ts_str)
                    print(alert_box)

                    try:
                        save_alert({
                            "transaction_id": tx_id,
                            "amount": amt,
                            "fraud_probability": prob,
                            "risk_level": risk_level,
                            "alert_status": "NEW"
                        })
                    except Exception as e:
                        print(f"[Consumer] Alert save error: {e}")
                else:
                    print(f"[Consumer] Processed tx: {tx_id} | Amt: ${amt:.2f} | Prob: {prob:.4f} | Risk: {risk_level}")

                processed_count += 1

                # 5. Dashboard Callback
                if callback:
                    callback({
                        **pred_result,
                        "timestamp": ts_str,
                        "actual_class": payload.get("Class", None)
                    })

                if max_records and processed_count >= max_records:
                    print(f"[Consumer] Reached target limit of {max_records} transactions.")
                    return processed_count

    except KeyboardInterrupt:
        print("\n[Consumer] Stopped by user.")
    finally:
        consumer.close()
        print(f"[Consumer] Terminated. Total processed: {processed_count}, Alerts triggered: {alerts_count}")

    return processed_count

if __name__ == "__main__":
    start_consuming_transactions(max_records=20)
