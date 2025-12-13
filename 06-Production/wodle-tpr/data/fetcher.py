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
        self.batch_size = config.get('performance', {}).get('batch_size', 10000)

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

        return pd.json_normalize(all_documents, sep='.')
