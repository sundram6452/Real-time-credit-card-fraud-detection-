"""
Real-time streaming module: Producer, Consumer, and Kafka-compatible simulator.
"""
from src.streaming.producer import start_producing_transactions, get_producer
from src.streaming.consumer import start_consuming_transactions, get_consumer, format_fraud_alert_box
from src.streaming.kafka_simulator import (
    SimulatedKafkaProducer,
    SimulatedKafkaConsumer,
    SimulatedKafkaBroker
)

__all__ = [
    "start_producing_transactions",
    "get_producer",
    "start_consuming_transactions",
    "get_consumer",
    "format_fraud_alert_box",
    "SimulatedKafkaProducer",
    "SimulatedKafkaConsumer",
    "SimulatedKafkaBroker"
]
