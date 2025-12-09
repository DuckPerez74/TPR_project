import pandas as pd
import warnings
from datetime import datetime
from opensearchpy import OpenSearch
from config import INDEX_PATTERN

warnings.filterwarnings('ignore', 'Unverified HTTPS request')

def fetch_all_logs_once(client: OpenSearch, start_time: datetime, end_time: datetime):
    query = {
        "query": {
            "range": {
                "timestamp": {
                    "gte": start_time.isoformat(),
                    "lt": end_time.isoformat()
                }
            }
        }
    }

    try:
        response = client.search(index=INDEX_PATTERN, body=query, size=10000, scroll="2m")
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']

        while scroll_id and len(response['hits']['hits']) > 0:
            response = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get('_scroll_id')
            hits.extend(response['hits']['hits'])

        if not hits:
            return pd.DataFrame()

        logs = [hit['_source'] for hit in hits]
        df = pd.json_normalize(logs, sep='.')

        if 'timestamp' in df.columns:
            df['@timestamp'] = pd.to_datetime(df['timestamp'])
        elif '@timestamp' in df.columns:
            df['@timestamp'] = pd.to_datetime(df['@timestamp'])
        else:
            return pd.DataFrame()

        if df['@timestamp'].dt.tz is None:
            df['@timestamp'] = df['@timestamp'].dt.tz_localize('UTC')

        return df

    except Exception:
        return pd.DataFrame()
