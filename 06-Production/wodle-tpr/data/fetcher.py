import pandas as pd
from datetime import datetime
from typing import Optional
import time
from opensearchpy.exceptions import TransportError


class LogFetcher:
    def __init__(self, client, config: dict):
        self.client = client
        self.index_pattern = config.get('indices', {}).get('raw_logs', {}).get('pattern', 'wazuh-alerts-4.x-*')
        self.location_filter = config.get('indices', {}).get('raw_logs', {}).get('location_filter')
        self.entity_field = config.get('indices', {}).get('raw_logs', {}).get('company_id_field', 'data.entities')
        self.batch_size = config.get('performance', {}).get('batch_size', 10000)
        self.max_logs_per_fetch = config.get('performance', {}).get('max_logs_per_fetch', 1000000)  # 1M default

    def count_logs(self, start_time: datetime, end_time: datetime) -> int:
        """Count logs in time range before fetching to detect large chunks"""
        must_filters = [
            {
                "range": {
                    "timestamp": {
                        "gte": start_time.isoformat(),
                        "lt": end_time.isoformat()
                    }
                }
            }
        ]

        if self.location_filter:
            must_filters.append({
                "match_phrase": {
                    "location": self.location_filter
                }
            })

        query = {
            "query": {
                "bool": {
                    "must": must_filters
                }
            }
        }

        try:
            response = self.client.count(index=self.index_pattern, body=query)
            return response.get('count', 0)
        except Exception as e:
            print(f"WARNING: Failed to count logs: {str(e)}")
            return 0

    def fetch_logs(self, start_time: datetime, end_time: datetime, max_retries: int = 3) -> pd.DataFrame:
        must_filters = [
            {
                "range": {
                    "timestamp": {
                        "gte": start_time.isoformat(),
                        "lt": end_time.isoformat()
                    }
                }
            }
        ]

        if self.location_filter:
            must_filters.append({
                "match_phrase": {
                    "location": self.location_filter
                }
            })

        query = {
            "query": {
                "bool": {
                    "must": must_filters
                }
            },
            "sort": [{"timestamp": "asc"}]
        }

        all_documents = []
        scroll_id = None
        retry_count = 0

        while retry_count <= max_retries:
            try:
                response = self.client.search(
                    index=self.index_pattern,
                    body=query,
                    scroll='5m',  # Increased from 2m to 5m
                    size=self.batch_size
                )

                scroll_id = response.get('_scroll_id')
                hits = response['hits']['hits']

                while hits:
                    all_documents.extend([hit['_source'] for hit in hits])

                    try:
                        response = self.client.scroll(
                            scroll_id=scroll_id,
                            scroll='5m'
                        )
                        hits = response['hits']['hits']
                        scroll_id = response.get('_scroll_id')

                    except TransportError as e:
                        if e.status_code == 429:  # Circuit breaker
                            print(f"WARNING: Circuit breaker hit during scroll. Waiting 30s before retry...")
                            time.sleep(30)
                            retry_count += 1
                            if retry_count > max_retries:
                                raise
                            continue
                        else:
                            raise

                # Success - break retry loop
                break

            except TransportError as e:
                if e.status_code == 429:  # Circuit breaker
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"ERROR: Max retries reached for circuit breaker. Giving up.")
                        raise

                    wait_time = 30 * retry_count  # Exponential backoff
                    print(f"WARNING: Circuit breaker triggered. Retry {retry_count}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)

                    # Clear scroll before retry
                    if scroll_id:
                        try:
                            self.client.clear_scroll(scroll_id=scroll_id)
                        except:
                            pass
                    scroll_id = None
                    all_documents = []  # Reset on retry
                else:
                    raise

            finally:
                if scroll_id:
                    try:
                        self.client.clear_scroll(scroll_id=scroll_id)
                    except Exception as e:
                        import sys
                        print(f"WARNING: Failed to clear scroll context {scroll_id}: {str(e)}", file=sys.stderr)

        if not all_documents:
            return pd.DataFrame()

        # Convert to DataFrame and immediately free the documents list
        df = pd.json_normalize(all_documents, sep='.')
        del all_documents

        # Force garbage collection to free memory from the large list
        import gc
        gc.collect()

        return df

    def get_active_entities(self, start_time: datetime, end_time: datetime) -> list:
        """
        Get list of active entities in time range using aggregation.
        Fast query that only returns entity IDs, not full logs.
        """
        must_filters = [
            {
                "range": {
                    "timestamp": {
                        "gte": start_time.isoformat(),
                        "lt": end_time.isoformat()
                    }
                }
            }
        ]

        if self.location_filter:
            must_filters.append({
                "match_phrase": {
                    "location": self.location_filter
                }
            })

        query = {
            "size": 0,  # Don't return documents, only aggregation
            "query": {
                "bool": {
                    "must": must_filters
                }
            },
            "aggs": {
                "active_entities": {
                    "terms": {
                        "field": self.entity_field,
                        "size": 5000,  # Support up to 5000 entities per chunk
                        "order": {"_count": "desc"}
                    }
                }
            }
        }

        try:
            response = self.client.search(index=self.index_pattern, body=query)
            buckets = response['aggregations']['active_entities']['buckets']

            # Filter out invalid entities ("-", "", null)
            entities = [
                bucket['key']
                for bucket in buckets
                if bucket['key'] not in ['-', '', None]
            ]

            return entities
        except Exception as e:
            print(f"WARNING: Failed to get active entities: {str(e)}")
            return []

    def fetch_logs_by_entity(self, entity_id: str, start_time: datetime, end_time: datetime, max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch logs for a specific entity in time range.
        Much more memory-efficient than fetching all entities at once.
        """
        must_filters = [
            {
                "range": {
                    "timestamp": {
                        "gte": start_time.isoformat(),
                        "lt": end_time.isoformat()
                    }
                }
            },
            {
                "term": {
                    self.entity_field: entity_id
                }
            }
        ]

        if self.location_filter:
            must_filters.append({
                "match_phrase": {
                    "location": self.location_filter
                }
            })

        query = {
            "query": {
                "bool": {
                    "must": must_filters
                }
            },
            "sort": [{"timestamp": "asc"}]
        }

        all_documents = []
        scroll_id = None
        retry_count = 0

        while retry_count <= max_retries:
            try:
                response = self.client.search(
                    index=self.index_pattern,
                    body=query,
                    scroll='5m',
                    size=self.batch_size
                )

                scroll_id = response.get('_scroll_id')
                hits = response['hits']['hits']

                while hits:
                    all_documents.extend([hit['_source'] for hit in hits])

                    try:
                        response = self.client.scroll(
                            scroll_id=scroll_id,
                            scroll='5m'
                        )
                        hits = response['hits']['hits']
                        scroll_id = response.get('_scroll_id')

                    except TransportError as e:
                        if e.status_code == 429:  # Circuit breaker
                            print(f"WARNING: Circuit breaker hit during scroll for entity {entity_id}. Waiting 30s...")
                            time.sleep(30)
                            retry_count += 1
                            if retry_count > max_retries:
                                raise
                            continue
                        else:
                            raise

                # Success - break retry loop
                break

            except TransportError as e:
                if e.status_code == 429:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"ERROR: Max retries reached for entity {entity_id}. Giving up.")
                        raise

                    wait_time = 30 * retry_count
                    print(f"WARNING: Circuit breaker for entity {entity_id}. Retry {retry_count}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)

                    if scroll_id:
                        try:
                            self.client.clear_scroll(scroll_id=scroll_id)
                        except:
                            pass
                    scroll_id = None
                    all_documents = []
                else:
                    raise

            finally:
                if scroll_id:
                    try:
                        self.client.clear_scroll(scroll_id=scroll_id)
                    except Exception as e:
                        import sys
                        print(f"WARNING: Failed to clear scroll context {scroll_id}: {str(e)}", file=sys.stderr)

        if not all_documents:
            return pd.DataFrame()

        # Convert to DataFrame and immediately free the documents list
        df = pd.json_normalize(all_documents, sep='.')
        del all_documents

        import gc
        gc.collect()

        return df
