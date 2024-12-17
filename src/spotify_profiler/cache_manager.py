import sqlite3
import json
import time
from datetime import datetime, timedelta
import pickle
from typing import Any, Optional, Dict
import logging

class CacheManager:
    def __init__(self, db_path: str = "music_cache.db"):
        self.db_path = db_path
        self._init_db()
        self.logger = logging.getLogger(__name__)

    def _init_db(self):
        """Initialize SQLite database with necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    source TEXT,
                    timestamp FLOAT,
                    expiry FLOAT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    api TEXT PRIMARY KEY,
                    last_request FLOAT,
                    remaining_calls INTEGER
                )
            """)

    def get(self, key: str, source: str) -> Optional[Any]:
        """Get item from cache if it exists and isn't expired"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT data, expiry FROM api_cache WHERE key = ? AND source = ?",
                    (key, source)
                ).fetchone()
                
                if result:
                    data, expiry = result
                    if expiry > time.time():
                        return pickle.loads(data)
                    else:
                        # Clean up expired entry
                        conn.execute(
                            "DELETE FROM api_cache WHERE key = ? AND source = ?",
                            (key, source)
                        )
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
        return None

    def set(self, key: str, data: Any, source: str, ttl: int = 86400):
        """Store item in cache with expiration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (key, data, source, timestamp, expiry) VALUES (?, ?, ?, ?, ?)",
                    (key, pickle.dumps(data), source, time.time(), time.time() + ttl)
                )
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")

    def check_rate_limit(self, api: str, limit: int, window: int = 60) -> bool:
        """Check if we're within rate limits"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT last_request, remaining_calls FROM rate_limits WHERE api = ?",
                    (api,)
                ).fetchone()
                
                current_time = time.time()
                
                if result:
                    last_request, remaining = result
                    # Reset if window has passed
                    if current_time - last_request > window:
                        remaining = limit
                    
                    if remaining > 0:
                        conn.execute(
                            "UPDATE rate_limits SET last_request = ?, remaining_calls = ?",
                            (current_time, remaining - 1)
                        )
                        return True
                    return False
                else:
                    # First request for this API
                    conn.execute(
                        "INSERT INTO rate_limits (api, last_request, remaining_calls) VALUES (?, ?, ?)",
                        (api, current_time, limit - 1)
                    )
                    return True
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return False
