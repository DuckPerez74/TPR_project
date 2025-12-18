from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from abc import ABC


class CacheManager(ABC):
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 60):
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None

        entry = self._cache[key]
        timestamp = entry.get('timestamp')

        # Check if expired
        if timestamp and (datetime.utcnow() - timestamp) > self.ttl:
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return entry.get('value')

    def set(self, key: str, value: Any) -> None:
        # Update existing entry
        if key in self._cache:
            self._cache.move_to_end(key)

        # Add new entry
        self._cache[key] = {
            'value': value,
            'timestamp': datetime.utcnow()
        }

        # Evict oldest if over max size
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (FIFO/LRU)

    def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_minutes': self.ttl.total_seconds() / 60,
            'keys': list(self._cache.keys())
        }

    def cleanup_expired(self) -> int:
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._cache.items()
            if (now - entry.get('timestamp', now)) > self.ttl
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (without expiry check)."""
        return key in self._cache
