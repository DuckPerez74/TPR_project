import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta, timezone
from opensearchpy import OpenSearch
from config import OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASSWORD

warnings.filterwarnings('ignore', 'Unverified HTTPS request')

def create_opensearch_client():
    client = OpenSearch(
        hosts=[OPENSEARCH_HOST],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    if not client.ping():
        raise ConnectionError("Failed to connect to OpenSearch")
    return client

def calculate_entropy(series):
    if series.empty:
        return 0
    counts = series.value_counts()
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def calculate_gini(series):
    if series.empty or series.sum() == 0:
        return 0
    sorted_series = series.sort_values().to_numpy()
    n = len(sorted_series)
    cumx = np.cumsum(sorted_series)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def clean_metrics(metrics):
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                clean[key] = 0
            else:
                clean[key] = value
        else:
            clean[key] = value
    return clean

def get_time_range(end_time: datetime, window_minutes: int):
    start_time = end_time - timedelta(minutes=window_minutes)
    return start_time, end_time
