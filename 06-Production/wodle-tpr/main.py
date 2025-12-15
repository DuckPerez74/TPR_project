#!/usr/bin/env python3
import sys
import os
import signal
import threading
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from core import ConfigLoader, Scheduler, OpenSearchClient, WazuhLogger
from data import LogFetcher, DataPreprocessor
from metrics import L1MetricsCalculator, L2MetricsCalculator, MetricsStorage
from detection import AnomalyDetector, HierarchicalAnalyzer
from utils import get_time_range


class TimeoutError(Exception):
    pass


class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.shutdown_requested = True

    def should_continue(self):
        return not self.shutdown_requested


def timeout_handler(timeout_seconds, shutdown_flag):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Execution timeout")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive() or shutdown_flag.shutdown_requested:
                raise TimeoutError(f"Execution exceeded {timeout_seconds}s timeout")

            if isinstance(result[0], Exception):
                raise result[0]

            return result[0]
        return wrapper
    return decorator


def process_entity(entity_id: str, entity_df, current_time: datetime,
                   scheduler, l1_calc, l2_calc, storage, detector, analyzer, logger,
                   detection_enabled: bool, shutdown_flag):

    if shutdown_flag.should_continue() is False:
        return False

    l1_metrics_by_window = {}
    l2_metrics_by_window = {}
    current_minute = current_time.minute

    window_data = {}
    for window in [60, 30, 10]:
        start_time, end_time = get_time_range(current_time, window)
        window_df = entity_df[(entity_df['@timestamp'] >= start_time) &
                             (entity_df['@timestamp'] < end_time)]
        if not window_df.empty:
            window_data[window] = window_df

    for window in [60, 30, 10]:
        if shutdown_flag.should_continue() is False:
            break

        window_df = window_data.get(window)
        if window_df is None:
            continue

        try:
            l1_metrics = l1_calc.calculate(window_df, window)
            l1_metrics_by_window[window] = l1_metrics

            if scheduler.should_save_metrics('L1', window, current_minute):
                storage.save_l1_metrics(entity_id, current_time, window, l1_metrics)
        except (ValueError, TypeError, KeyError) as e:
            logger.log_error(f"L1 metrics calculation failed for entity {entity_id}, window {window}", e)
            continue

        l2_enabled = l1_calc.config.get('metrics', {}).get('layers', {}).get('L2', {}).get('enabled', False)
        if l2_enabled:
            try:
                l2_results = []
                for dimension in l2_calc.dimensions:
                    dim_results = l2_calc.calculate(window_df, dimension)
                    l2_results.extend(dim_results)

                l2_metrics_by_window[window] = l2_results

                if scheduler.should_save_metrics('L2', window, current_minute):
                    storage.save_l2_metrics(entity_id, current_time, window, l2_results)
            except (ValueError, TypeError, KeyError) as e:
                logger.log_error(f"L2 metrics calculation failed for entity {entity_id}, window {window}", e)

    if not detection_enabled:
        return False

    if 60 not in l1_metrics_by_window and 60 not in l2_metrics_by_window:
        return False

    try:
        result = analyzer.analyze(detector, entity_id, l1_metrics_by_window, l2_metrics_by_window)

        if result is not None:
            logger.log_anomaly_alert(entity_id, result)
            return True
    except (RuntimeError, ValueError, KeyError) as e:
        logger.log_error(f"Anomaly detection failed for entity {entity_id}", e)

    return False


def main():
    load_dotenv()

    shutdown_flag = GracefulShutdown()
    current_time = datetime.now(timezone.utc)
    logger = None

    try:
        config = ConfigLoader().get_all()

        observation_windows = config.get('observation_windows', {}).get('enabled', [60])
        if not observation_windows or not isinstance(observation_windows, list):
            print("ERROR: Invalid observation_windows configuration", file=sys.stderr)
            sys.exit(1)

        max_window = max(observation_windows)
        max_execution_time = config.get('performance', {}).get('max_execution_time_seconds', 300)

        scheduler = Scheduler(config)

        if not scheduler.should_execute(current_time):
            sys.exit(0)

        client = OpenSearchClient.get_instance(config)
        logger = WazuhLogger(config)

        fetcher = LogFetcher(client, config)
        preprocessor = DataPreprocessor(config)

        l1_calc = L1MetricsCalculator(config)
        l2_calc = L2MetricsCalculator(config)
        storage = MetricsStorage(client, config)

        detector = AnomalyDetector(config)
        analyzer = HierarchicalAnalyzer(config)

        detection_enabled = config.get('detection', {}).get('enabled', True)

        if not detector.model_loader.has_any_models_on_disk():
            logger.log_error("No models found. Run warmup.py first to collect metrics, then train.py to train models.", None)
            sys.exit(1)

        start_time, end_time = get_time_range(current_time, max_window)
        start_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Optimization: Get active entities first, then fetch logs per entity
        # This prevents loading massive datasets into memory
        active_entities = fetcher.get_active_entities(start_time, end_time)

        if not active_entities:
            logger.log_error("No active entities found in time range", None)
            sys.exit(0)

        anomalies_found = 0
        processed_entities = 0

        for entity_id in active_entities:
            if shutdown_flag.should_continue() is False:
                logger.log_error("Shutdown requested, stopping entity processing", None)
                break

            elapsed_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - start_ms
            if elapsed_ms > (max_execution_time * 1000):
                logger.log_error(f"Max execution time ({max_execution_time}s) exceeded", None)
                break

            # Fetch only this entity's logs
            entity_df = fetcher.fetch_logs_by_entity(entity_id, start_time, end_time)
            
            if entity_df.empty:
                continue

            # Preprocess specific entity data
            entity_df = preprocessor.prepare(entity_df)

            if process_entity(entity_id, entity_df, current_time, scheduler,
                            l1_calc, l2_calc, storage, detector, analyzer, logger,
                            detection_enabled, shutdown_flag):
                anomalies_found += 1
            
            processed_entities += 1
            
            # Explicit cleanup
            del entity_df

        # Flush any remaining metrics in buffer
        storage.flush_metrics()

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        execution_time = end_ms - start_ms

        if config.get('features', {}).get('log_detection_runs', True):
            logger.log_detection_run(processed_entities, anomalies_found, execution_time)


    except (KeyError, ValueError, TypeError) as e:
        if logger:
            logger.log_error(f"Configuration or data error: {str(e)}", e)
        else:
            print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except (ConnectionError, TimeoutError) as e:
        if logger:
            logger.log_error(f"Connection error: {str(e)}", e)
        else:
            print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.log_error(f"Unexpected error: {str(e)}", e)
        else:
            print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
