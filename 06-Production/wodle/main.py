#!/usr/bin/env python3
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timezone
from scheduler import get_execution_windows
from utils import create_opensearch_client, get_time_range
from data_fetcher import fetch_all_logs_once
from metrics_calculator import calculate_l1_metrics, save_metrics_to_index
from anomaly_detector import AnomalyDetector, analyze_entity_windows
from wazuh_logger import WazuhAnomalyLogger
from config import OBSERVATION_WINDOWS, HISTORICAL_WINDOW, COMPANY_ID_FIELD

def process_entity(client, detector, logger, entity_id, entity_df, current_time):
    metrics_by_window = {}
    current_minute = current_time.minute

    start_time_10, end_time_10 = get_time_range(current_time, HISTORICAL_WINDOW)
    df_10 = entity_df[(entity_df['@timestamp'] >= start_time_10) & (entity_df['@timestamp'] < end_time_10)]

    if df_10.empty:
        return False

    metrics_10 = calculate_l1_metrics(df_10, HISTORICAL_WINDOW)
    save_metrics_to_index(client, entity_id, current_time, metrics_10, HISTORICAL_WINDOW)
    metrics_by_window[10] = metrics_10

    for window in OBSERVATION_WINDOWS:
        if window == 10:
            continue

        start_time, end_time = get_time_range(current_time, window)
        df = entity_df[(entity_df['@timestamp'] >= start_time) & (entity_df['@timestamp'] < end_time)]

        if not df.empty:
            metrics = calculate_l1_metrics(df, window)
            metrics_by_window[window] = metrics

            should_save = False
            if window == 60 and current_minute == 0:
                should_save = True
            elif window == 30 and current_minute in [0, 30]:
                should_save = True

            if should_save:
                save_metrics_to_index(client, entity_id, current_time, metrics, window)

    if 60 not in metrics_by_window or 30 not in metrics_by_window:
        return False

    result = analyze_entity_windows(
        detector,
        entity_id,
        metrics_by_window.get(60, {}),
        metrics_by_window.get(30, {}),
        metrics_by_window.get(10, {})
    )

    if result is not None:
        selected_window = result['selected_window']
        score = result['results'][str(selected_window)]['score']
        model_used = result.get('model_used')
        cluster_id = result.get('cluster_id')

        metrics_summary = {
            'total_requests': metrics_by_window[selected_window].get('total_requests', 0),
            'error_rate': metrics_by_window[selected_window].get('error_rate', 0),
            'mean_response_time': metrics_by_window[selected_window].get('mean_response_time', 0),
            'model_used': model_used,
            'cluster_id': cluster_id
        }

        logger.log_anomaly(entity_id, selected_window, score, metrics_summary)
        return True

    return False

def main():
    current_time = datetime.now(timezone.utc)

    execution_windows = get_execution_windows(current_time)
    if not execution_windows:
        sys.exit(0)

    try:
        client = create_opensearch_client()
        detector = AnomalyDetector()
        logger = WazuhAnomalyLogger()

        start_time, end_time = get_time_range(current_time, max(OBSERVATION_WINDOWS))
        all_data = fetch_all_logs_once(client, start_time, end_time)

        if all_data.empty:
            sys.exit(0)

        if COMPANY_ID_FIELD not in all_data.columns:
            sys.exit(0)

        active_entities = all_data[COMPANY_ID_FIELD].dropna().unique()
        active_entities = [str(e).strip() for e in active_entities if str(e).strip() not in ["-", ""]]

        if not active_entities:
            sys.exit(0)

        anomalies_found = 0
        for entity_id in active_entities:
            entity_df = all_data[all_data[COMPANY_ID_FIELD] == entity_id]
            if process_entity(client, detector, logger, entity_id, entity_df, current_time):
                anomalies_found += 1

        logger.log_detection_run(len(active_entities), anomalies_found)

    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
