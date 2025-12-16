import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import OrderedDict
import sys
from constants import L1_FEATURE_ORDER as FEATURE_ORDER


class WindowBuffer:
    """
    Manages historical window data for entities to support aggregated cluster prediction.

    This class fetches and caches recent metric windows from OpenSearch, maintaining
    consistency with the training approach (using mean aggregation over multiple windows).
    """

    def __init__(self, opensearch_client, config: dict):
        """
        Initialize WindowBuffer.

        Args:
            opensearch_client: OpenSearch client instance
            config: Configuration dictionary with cluster_prediction settings
        """
        self.client = opensearch_client

        cluster_config = config.get('detection', {}).get('cluster_prediction', {})

        self.enabled = cluster_config.get('enabled', True)
        self.lookback_days = cluster_config.get('lookback_days', 7)
        self.min_windows_required = cluster_config.get('min_windows_required', 24)
        self.cache_enabled = cluster_config.get('cache_enabled', True)
        self.cache_ttl_minutes = cluster_config.get('cache_ttl_minutes', 60)
        self.cache_max_entities = cluster_config.get('cache_max_entities', 1000)

        self.metrics_index = config.get('indices', {}).get('metrics', {}).get('name', 'metrics-tpr')

        # Cache: entity_id -> {'windows': np.array, 'timestamp': datetime, 'count': int}
        self._cache = OrderedDict()

        print(f"[WindowBuffer] Initialized with lookback_days={self.lookback_days}, "
              f"min_windows={self.min_windows_required}, cache_enabled={self.cache_enabled}",
              file=sys.stderr)

    def get_recent_windows(self, entity_id: str, observation_window: int = 60) -> Optional[np.ndarray]:
        """
        Get recent metric windows for an entity.

        Args:
            entity_id: Entity identifier
            observation_window: Window size in minutes (default: 60)

        Returns:
            np.ndarray of shape (n_windows, n_features) or None if insufficient data
        """
        if not self.enabled:
            return None

        # Check cache first
        if self.cache_enabled:
            cached = self._get_from_cache(entity_id)
            if cached is not None:
                return cached

        # Fetch from OpenSearch
        windows = self._fetch_from_opensearch(entity_id, observation_window)

        if windows is None or len(windows) < self.min_windows_required:
            print(f"[WindowBuffer] Insufficient windows for {entity_id}: "
                  f"found {len(windows) if windows is not None else 0}, "
                  f"required {self.min_windows_required}",
                  file=sys.stderr)
            return None

        # Cache the result
        if self.cache_enabled:
            self._add_to_cache(entity_id, windows)

        return windows

    def _get_from_cache(self, entity_id: str) -> Optional[np.ndarray]:
        """Get windows from cache if valid."""
        if entity_id not in self._cache:
            return None

        cache_entry = self._cache[entity_id]
        cache_age = datetime.utcnow() - cache_entry['timestamp']

        if cache_age.total_seconds() > self.cache_ttl_minutes * 60:
            # Cache expired
            del self._cache[entity_id]
            return None

        # Move to end (LRU)
        self._cache.move_to_end(entity_id)
        return cache_entry['windows']

    def _add_to_cache(self, entity_id: str, windows: np.ndarray):
        """Add windows to cache with LRU eviction."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.cache_max_entities:
            self._cache.popitem(last=False)  # Remove oldest

        self._cache[entity_id] = {
            'windows': windows,
            'timestamp': datetime.utcnow(),
            'count': len(windows)
        }

    def _fetch_from_opensearch(self, entity_id: str, observation_window: int) -> Optional[np.ndarray]:
        """
        Fetch recent windows from OpenSearch.

        Args:
            entity_id: Entity identifier
            observation_window: Window size in minutes

        Returns:
            np.ndarray of shape (n_windows, n_features) or None
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=self.lookback_days)

            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "entity_id.keyword": entity_id
                                }
                            },
                            {
                                "term": {
                                    "layer": "L1"
                                }
                            },
                            {
                                "term": {
                                    "observation_window": observation_window
                                }
                            },
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": start_time.isoformat(),
                                        "lt": end_time.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 10000,
                "sort": [
                    {
                        "@timestamp": {
                            "order": "desc"
                        }
                    }
                ]
            }

            response = self.client.search(
                index=self.metrics_index,
                body=query,
                scroll='2m'
            )

            hits = response['hits']['hits']
            all_docs = hits.copy()

            scroll_id = response.get('_scroll_id')
            while scroll_id and len(hits) > 0 and len(all_docs) < 10000:
                response = self.client.scroll(scroll_id=scroll_id, scroll='2m')
                scroll_id = response.get('_scroll_id')
                hits = response['hits']['hits']
                all_docs.extend(hits)

            if scroll_id:
                try:
                    self.client.clear_scroll(scroll_id=scroll_id)
                except Exception:
                    pass

            if not all_docs:
                return None

            # Convert to feature vectors
            windows = []
            for doc in all_docs:
                source = doc['_source']
                metrics = source.get('metrics', {})

                vector = []
                for feature in FEATURE_ORDER:
                    vector.append(metrics.get(feature, 0))

                windows.append(vector)

            if not windows:
                return None

            return np.array(windows)

        except Exception as e:
            print(f"[WindowBuffer] Error fetching windows for {entity_id}: {str(e)}", file=sys.stderr)
            return None

    def clear_cache(self, entity_id: Optional[str] = None):
        """
        Clear cache for a specific entity or all entities.

        Args:
            entity_id: Specific entity to clear, or None to clear all
        """
        if entity_id is None:
            self._cache.clear()
        elif entity_id in self._cache:
            del self._cache[entity_id]

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_windows = sum(entry['count'] for entry in self._cache.values())

        return {
            'cached_entities': len(self._cache),
            'total_windows': total_windows,
            'max_entities': self.cache_max_entities,
            'ttl_minutes': self.cache_ttl_minutes,
            'enabled': self.cache_enabled
        }
