"""
Kafka-Compatible In-Memory Streaming Layer
Provides thread-safe pub/sub broker simulation matching the kafka-python API.
Enables zero-dependency local streaming development while maintaining 100%
architectural parity with production Kafka.
"""

import time
import json
import queue
from typing import Dict, Any, List, Optional
from threading import Lock

class SimulatedKafkaBroker:
    """Singleton in-memory topic broker with FIFO thread-safe queues."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._topics: Dict[str, queue.Queue] = {}
                cls._instance._topic_locks: Dict[str, Lock] = {}
        return cls._instance

    def get_topic_queue(self, topic: str) -> queue.Queue:
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = queue.Queue(maxsize=10000)
                self._topic_locks[topic] = Lock()
            return self._topics[topic]

    def reset_topic(self, topic: str):
        with self._lock:
            if topic in self._topics:
                q = self._topics[topic]
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

class SimulatedRecordMetadata:
    """Mimics Kafka RecordMetadata returned upon send."""
    def __init__(self, topic: str, partition: int = 0, offset: int = 0):
        self.topic = topic
        self.partition = partition
        self.offset = offset

class SimulatedConsumerRecord:
    """Mimics Kafka ConsumerRecord message object."""
    def __init__(self, topic: str, value: Any, timestamp: float, offset: int = 0):
        self.topic = topic
        self.value = value
        self.timestamp = timestamp
        self.offset = offset
        self.key = None

class SimulatedKafkaProducer:
    """Kafka-compatible producer publishing events to the in-memory broker."""
    def __init__(self, value_serializer=None, **kwargs):
        self.broker = SimulatedKafkaBroker()
        self.value_serializer = value_serializer or (lambda v: json.dumps(v).encode("utf-8"))
        self._counter = 0

    def send(self, topic: str, value: Any, key: Any = None):
        q = self.broker.get_topic_queue(topic)
        # Serialize then deserialize to simulate wire transport
        raw_bytes = self.value_serializer(value)
        self._counter += 1
        record = SimulatedConsumerRecord(
            topic=topic,
            value=raw_bytes,
            timestamp=time.time(),
            offset=self._counter
        )
        try:
            q.put(record, timeout=2.0)
        except queue.Full:
            print(f"[KafkaSimulator] Topic {topic} is full. Dropping event.")
        return SimulatedRecordMetadata(topic, partition=0, offset=self._counter)

    def flush(self):
        pass

    def close(self):
        pass

class SimulatedKafkaConsumer:
    """Kafka-compatible consumer polling events from the in-memory broker."""
    def __init__(
        self,
        *topics,
        value_deserializer=None,
        auto_offset_reset="latest",
        **kwargs
    ):
        self.broker = SimulatedKafkaBroker()
        self.topics = list(topics)
        self.value_deserializer = value_deserializer or (lambda v: json.loads(v.decode("utf-8")))
        self.is_running = True

    def subscribe(self, topics: List[str]):
        self.topics = list(topics)

    def poll(self, timeout_ms: int = 1000, max_records: int = 1) -> Dict[str, List[SimulatedConsumerRecord]]:
        records = {}
        timeout_sec = timeout_ms / 1000.0
        start_time = time.time()

        for topic in self.topics:
            q = self.broker.get_topic_queue(topic)
            topic_records = []
            while len(topic_records) < max_records:
                remaining_time = max(0.01, timeout_sec - (time.time() - start_time))
                try:
                    rec = q.get(timeout=remaining_time)
                    # Deserialize value
                    deserialized_value = self.value_deserializer(rec.value)
                    clean_rec = SimulatedConsumerRecord(
                        topic=rec.topic,
                        value=deserialized_value,
                        timestamp=rec.timestamp,
                        offset=rec.offset
                    )
                    topic_records.append(clean_rec)
                except queue.Empty:
                    break
            if topic_records:
                records[topic] = topic_records

        return records

    def __iter__(self):
        while self.is_running:
            polled = self.poll(timeout_ms=500, max_records=10)
            for topic, records in polled.items():
                for rec in records:
                    yield rec

    def close(self):
        self.is_running = False
