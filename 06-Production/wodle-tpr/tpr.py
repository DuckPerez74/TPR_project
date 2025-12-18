#!/usr/bin/env python3
import sys
import signal
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from core import ConfigLoader, Scheduler, OpenSearchClient, WazuhLogger
from data import LogFetcher, DataPreprocessor
from metrics import L1MetricsCalculator, L2MetricsCalculator, MetricsStorage
from detection import AnomalyDetector, HierarchicalAnalyzer
from utils import get_time_range
from llm import LLMAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='TPR Anomaly Detection Wodle')
    parser.add_argument(
        '--force-minute',
        type=int,
        choices=range(0, 60),
        metavar='[0-59]',
        help='Force execution as if current minute is this value (for testing)'
    )
    return parser.parse_args()


class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.shutdown_requested = True

    def should_continue(self):
        return not self.shutdown_requested


def process_entity(entity_id: str, entity_df, current_time: datetime,
                   scheduler, l1_calc, l2_calc, storage, detector, analyzer, logger,
                   detection_enabled: bool, shutdown_flag, llm_analyzer=None,
                   scheduling_minute: int = None):
    from core.entity_processor import EntityProcessor

    processor = EntityProcessor(
        scheduler, l1_calc, l2_calc, storage, detector,
        analyzer, logger, llm_analyzer
    )

    return processor.process(
        entity_id, entity_df, current_time,
        shutdown_flag, detection_enabled, scheduling_minute
    )


def main():
    load_dotenv()
    args = parse_args()

    shutdown_flag = GracefulShutdown()
    current_time = datetime.now(timezone.utc)
    
    # For testing: create a time with overridden minute for scheduling decisions
    if args.force_minute is not None:
        scheduling_time = current_time.replace(minute=args.force_minute, second=0, microsecond=0)
        print(f"[TEST MODE] Forcing minute to :{args.force_minute:02d} for scheduling (real time: :{current_time.minute:02d})")
    else:
        scheduling_time = current_time
    
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

        if not scheduler.should_execute(scheduling_time):
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
        llm_analyzer = LLMAnalyzer()

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
                            detection_enabled, shutdown_flag, llm_analyzer,
                            scheduling_minute=scheduling_time.minute):
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
