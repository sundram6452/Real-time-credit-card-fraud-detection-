"""
Unified CLI Runner for Real-Time Credit Card Fraud Detection & Analytics System
"""

import argparse
import sys
import subprocess
import threading
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def cmd_generate_sample(args):
    """Generate synthetic sample dataset."""
    from data.sample.generate_sample import generate_sample_dataset
    from src.config import SAMPLE_DATA_PATH
    generate_sample_dataset(
        n_samples=args.samples,
        fraud_ratio=args.fraud_ratio,
        output_path=str(SAMPLE_DATA_PATH)
    )

def cmd_clean(args):
    """Execute data cleaning & audit pipeline."""
    from src.data.loader import load_transaction_data
    from src.data.cleaner import clean_transaction_data
    from src.config import PROCESSED_DATA_PATH
    df_raw = load_transaction_data(source="raw")
    clean_transaction_data(df_raw, remove_duplicates=True, output_path=str(PROCESSED_DATA_PATH))

def cmd_train(args):
    """Train baseline Logistic Regression and Random Forest models."""
    from src.models.train import train_fraud_detection_models
    train_fraud_detection_models(data_source=args.source)

def cmd_init_db(args):
    """Initialize database tables and optionally populate initial rows."""
    from src.database.connection import init_db
    from src.database.queries import populate_database_from_csv
    from src.config import PROCESSED_DATA_PATH
    init_db(drop_first=args.reset)
    if args.populate and PROCESSED_DATA_PATH.exists():
        populate_database_from_csv(str(PROCESSED_DATA_PATH), max_rows=args.rows)

def cmd_spark(args):
    """Execute PySpark / batch big data processing."""
    from src.bigdata.spark_analytics import run_spark_fraud_analytics
    from src.config import PROCESSED_DATA_PATH
    run_spark_fraud_analytics(str(PROCESSED_DATA_PATH))

def cmd_stream(args):
    """Run concurrent transaction streaming producer and fraud detection consumer."""
    from src.streaming.consumer import start_consuming_transactions
    from src.streaming.producer import start_producing_transactions

    print(f"Starting stream pipeline (Count: {args.count}, Delay: {args.delay}s, Kafka: {args.use_kafka})...")
    
    t_consumer = threading.Thread(
        target=start_consuming_transactions,
        kwargs={
            "max_records": args.count,
            "use_kafka": args.use_kafka,
            "threshold": args.threshold
        },
        daemon=True
    )
    t_consumer.start()
    time.sleep(0.5)

    start_producing_transactions(
        max_records=args.count,
        delay=args.delay,
        use_kafka=args.use_kafka,
        interleave_fraud_ratio=args.fraud_ratio
    )
    t_consumer.join(timeout=15)
    print("Streaming pipeline session finished.")

def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    print("Launching Streamlit Analytics Dashboard on port 8501...")
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(args.port)]
    subprocess.run(cmd)

def cmd_test(args):
    """Run pytest suite."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Real-Time Credit Card Fraud Detection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # generate-sample
    p_sample = subparsers.add_parser("generate-sample", help="Generate synthetic sample dataset")
    p_sample.add_argument("--samples", type=int, default=5000)
    p_sample.add_argument("--fraud-ratio", type=float, default=0.02)

    # clean
    subparsers.add_parser("clean", help="Clean raw dataset and generate audit log")

    # train
    p_train = subparsers.add_parser("train", help="Train ML models and save artifacts")
    p_train.add_argument("--source", type=str, default="auto", choices=["auto", "processed", "raw", "sample"])

    # init-db
    p_db = subparsers.add_parser("init-db", help="Initialize relational database")
    p_db.add_argument("--reset", action="store_true", help="Drop existing tables first")
    p_db.add_argument("--populate", action="store_true", help="Populate historical transactions")
    p_db.add_argument("--rows", type=int, default=15000)

    # spark-analytics
    subparsers.add_parser("spark-analytics", help="Run PySpark / batch big data analytics")

    # stream
    p_stream = subparsers.add_parser("stream", help="Run real-time streaming simulation")
    p_stream.add_argument("--count", type=int, default=30, help="Total transactions to stream")
    p_stream.add_argument("--delay", type=float, default=0.1, help="Delay between events in seconds")
    p_stream.add_argument("--threshold", type=float, default=0.30, help="Classification decision threshold")
    p_stream.add_argument("--fraud-ratio", type=float, default=0.20, help="Ratio of fraud events injected")
    p_stream.add_argument("--use-kafka", action="store_true", help="Use real Apache Kafka instead of simulator")

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Start interactive Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)

    # test
    subparsers.add_parser("test", help="Run test suite with pytest")

    args = parser.parse_args()

    dispatch = {
        "generate-sample": cmd_generate_sample,
        "clean": cmd_clean,
        "train": cmd_train,
        "init-db": cmd_init_db,
        "spark-analytics": cmd_spark,
        "stream": cmd_stream,
        "dashboard": cmd_dashboard,
        "test": cmd_test
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
