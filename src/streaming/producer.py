"""
Real-Time Transaction Stream Producer
Ingests transactions from dataset and produces JSON events to Kafka or Simulated Broker.
"""

import time
import json
import uuid
from typing import Optional, Generator, Dict, Any
import pandas as pd

from src.config import (
    USE_KAFKA,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    STREAM_DELAY
)
from src.data.loader import load_transaction_data
from src.streaming.kafka_simulator import SimulatedKafkaProducer

def get_producer(use_kafka: bool = USE_KAFKA):
    """Factory returning either actual KafkaProducer or SimulatedKafkaProducer."""
    if use_kafka:
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=5000
            )
            print(f"[Producer] Connected to production Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer, "kafka"
        except Exception as e:
            print(f"[Producer] Kafka connection failed ({e}). Falling back to Kafka-compatible simulated streaming layer.")
    
    producer = SimulatedKafkaProducer(
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    print("[Producer] Initialized Kafka-compatible simulated streaming producer.")
    return producer, "simulator"

def start_producing_transactions(
    topic: str = KAFKA_TOPIC,
    delay: float = STREAM_DELAY,
    max_records: Optional[int] = None,
    use_kafka: bool = USE_KAFKA,
    interleave_fraud_ratio: Optional[float] = 0.10,
    callback=None
) -> int:
    """
    Streams transactions one-by-one with configurable delay.
    Can interleave fraudulent transactions so stream demonstrations showcase alerts.
    """
    producer, mode = get_producer(use_kafka=use_kafka)
    
    # Load dataset
    df = load_transaction_data(source="auto", clean=True)
    
    if interleave_fraud_ratio and "Class" in df.columns:
        # Separate fraud and legit to interleave them intentionally for interactive demonstration
        df_fraud = df[df["Class"] == 1].sample(frac=1.0, random_state=42)
        df_legit = df[df["Class"] == 0].sample(frac=1.0, random_state=42)
        
        records_to_send = []
        n_fraud_total = len(df_fraud)
        n_legit_total = len(df_legit)
        f_idx, l_idx = 0, 0
        
        total_target = max_records if max_records else (n_fraud_total + n_legit_total)
        for i in range(total_target):
            # Periodically inject a fraud record based on interleave_fraud_ratio
            if (i % int(1.0 / interleave_fraud_ratio) == 0 or l_idx >= n_legit_total) and f_idx < n_fraud_total:
                records_to_send.append(df_fraud.iloc[f_idx].to_dict())
                f_idx += 1
            elif l_idx < n_legit_total:
                records_to_send.append(df_legit.iloc[l_idx].to_dict())
                l_idx += 1
            else:
                break
    else:
        records_to_send = df.to_dict(orient="records")
        if max_records:
            records_to_send = records_to_send[:max_records]

    sent_count = 0
    print(f"[Producer] Streaming {len(records_to_send)} transactions to topic '{topic}' (delay={delay}s, mode={mode})...")

    try:
        for idx, row in enumerate(records_to_send):
            tx_id = f"tx_stream_{uuid.uuid4().hex[:8]}"
            payload = {
                "transaction_id": tx_id,
                "Time": float(row["Time"]),
                "Amount": float(row["Amount"]),
                "Class": int(row.get("Class", 0)),
                "stream_timestamp": time.time()
            }
            # Add PCA features
            for i in range(1, 29):
                payload[f"V{i}"] = float(row[f"V{i}"])

            producer.send(topic, value=payload)
            sent_count += 1

            if callback:
                callback(payload)

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[Producer] Sent tx {sent_count}/{len(records_to_send)} (ID: {tx_id}, Amt: ${payload['Amount']:.2f}, Class: {payload['Class']})")

            time.sleep(delay)
    except KeyboardInterrupt:
        print("\n[Producer] Interrupted by user.")
    finally:
        producer.flush()
        producer.close()
        print(f"[Producer] Completed. Total transactions published: {sent_count}")

    return sent_count

if __name__ == "__main__":
    start_producing_transactions(max_records=50, delay=0.2)
