"""
Kafka-Compatible Simulated Streaming Layer
Provides thread-safe and multi-process pub/sub event queuing matching the kafka-python API.
Uses a lightweight local SQLite WAL queue so events can seamlessly flow between
different terminal processes (producer in one process, consumer in another) as well as within Streamlit.
"""

import time
import json
import sqlite3
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock
from src.config import DATA_DIR

QUEUE_DB_PATH = DATA_DIR / "stream_queue.db"

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

class SimulatedKafkaBroker:
    """
    Process-safe and thread-safe event queue.
    Uses SQLite with Write-Ahead Logging (WAL) to enable multi-process communication.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_storage()
        return cls._instance

    def _init_storage(self):
        """Initializes the SQLite queue table."""
        QUEUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(QUEUE_DB_PATH), timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stream_events (
                    offset INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic_offset ON stream_events(topic, offset);")
            conn.commit()

    def publish_event(self, topic: str, raw_payload: bytes) -> int:
        """Inserts an event into the topic queue and returns the auto-increment offset."""
        with sqlite3.connect(str(QUEUE_DB_PATH), timeout=10.0) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO stream_events (topic, payload, created_at) VALUES (?, ?, ?)",
                (topic, raw_payload, time.time())
            )
            offset = cur.lastrowid
            conn.commit()
            return offset

    def fetch_events(self, topic: str, after_offset: int, limit: int = 10) -> List[SimulatedConsumerRecord]:
        """Fetches events from the topic queue that have offset > after_offset."""
        records = []
        with sqlite3.connect(str(QUEUE_DB_PATH), timeout=10.0) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT offset, payload, created_at FROM stream_events WHERE topic = ? AND offset > ? ORDER BY offset ASC LIMIT ?",
                (topic, after_offset, limit)
            )
            rows = cur.fetchall()
            for offset, payload, created_at in rows:
                records.append(SimulatedConsumerRecord(
                    topic=topic,
                    value=payload,
                    timestamp=created_at,
                    offset=offset
                ))
        return records

    def reset_topic(self, topic: str):
        """Deletes all events for a given topic."""
        with sqlite3.connect(str(QUEUE_DB_PATH), timeout=10.0) as conn:
            conn.execute("DELETE FROM stream_events WHERE topic = ?", (topic,))
            conn.commit()

    def get_latest_offset(self, topic: str) -> int:
        """Returns the current highest offset for a topic."""
        with sqlite3.connect(str(QUEUE_DB_PATH), timeout=10.0) as conn:
            cur = conn.cursor()
            cur.execute("SELECT MAX(offset) FROM stream_events WHERE topic = ?", (topic,))
            row = cur.fetchone()
            return row[0] if (row and row[0] is not None) else 0

class SimulatedKafkaProducer:
    """Kafka-compatible producer publishing events to the shared queue."""
    def __init__(self, value_serializer=None, **kwargs):
        self.broker = SimulatedKafkaBroker()
        self.value_serializer = value_serializer or (lambda v: json.dumps(v).encode("utf-8"))

    def send(self, topic: str, value: Any, key: Any = None):
        raw_bytes = self.value_serializer(value)
        offset = self.broker.publish_event(topic, raw_bytes)
        return SimulatedRecordMetadata(topic, partition=0, offset=offset)

    def flush(self):
        pass

    def close(self):
        pass

class SimulatedKafkaConsumer:
    """Kafka-compatible consumer polling events from the shared queue."""
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
        
        # Track offset per topic
        self._offsets: Dict[str, int] = {}
        for topic in self.topics:
            if auto_offset_reset == "earliest":
                self._offsets[topic] = 0
            else:
                self._offsets[topic] = self.broker.get_latest_offset(topic)

    def subscribe(self, topics: List[str]):
        self.topics = list(topics)
        for topic in self.topics:
            if topic not in self._offsets:
                self._offsets[topic] = self.broker.get_latest_offset(topic)

    def poll(self, timeout_ms: int = 1000, max_records: int = 10) -> Dict[str, List[SimulatedConsumerRecord]]:
        records = {}
        timeout_sec = timeout_ms / 1000.0
        start_time = time.time()

        while (time.time() - start_time) < timeout_sec:
            found_any = False
            for topic in self.topics:
                current_off = self._offsets.get(topic, 0)
                fetched = self.broker.fetch_events(topic, after_offset=current_off, limit=max_records)
                
                if fetched:
                    found_any = True
                    clean_records = []
                    for rec in fetched:
                        self._offsets[topic] = max(self._offsets[topic], rec.offset)
                        val = self.value_deserializer(rec.value)
                        clean_records.append(SimulatedConsumerRecord(
                            topic=rec.topic,
                            value=val,
                            timestamp=rec.timestamp,
                            offset=rec.offset
                        ))
                    records[topic] = clean_records

            if found_any:
                break
            time.sleep(0.05)

        return records

    def __iter__(self):
        while self.is_running:
            polled = self.poll(timeout_ms=500, max_records=10)
            for topic, records in polled.items():
                for rec in records:
                    yield rec

    def close(self):
        self.is_running = False
