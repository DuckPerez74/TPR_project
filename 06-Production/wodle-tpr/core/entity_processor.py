"""
Entity processor for metrics calculation, detection, and analysis.

This module simplifies the main processing loop by extracting entity-level
operations into a dedicated class with clear separation of concerns.
"""
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from utils import get_time_range


class EntityProcessor:
    """
    Handles processing of a single entity: metrics calculation, detection, and LLM analysis.

    This class encapsulates the multi-step processing pipeline to keep main.py cleaner
    and improve testability.
    """

    def __init__(self, scheduler, l1_calc, l2_calc, storage, detector,
                 analyzer, logger, llm_analyzer=None):
        """
        Initialize entity processor with required components.

        Args:
            scheduler: Scheduler instance for timing decisions
            l1_calc: L1 metrics calculator
            l2_calc: L2 metrics calculator
            storage: Metrics storage
            detector: Anomaly detector
            analyzer: Hierarchical analyzer
            logger: Wazuh logger
            llm_analyzer: Optional LLM analyzer
        """
        self.scheduler = scheduler
        self.l1_calc = l1_calc
        self.l2_calc = l2_calc
        self.storage = storage
        self.detector = detector
        self.analyzer = analyzer
        self.logger = logger
        self.llm_analyzer = llm_analyzer

        # Check if L2 is enabled
        self.l2_enabled = l1_calc.config.get('metrics', {}).get('layers', {}).get('L2', {}).get('enabled', False)

    def process(self, entity_id: str, entity_df: pd.DataFrame, current_time: datetime,
                shutdown_flag, detection_enabled: bool, scheduling_minute: int = None) -> bool:
        """
        Process entity: calculate metrics, detect anomalies, run LLM analysis.

        Args:
            entity_id: Entity identifier
            entity_df: DataFrame with entity logs
            current_time: Current processing time
            shutdown_flag: Graceful shutdown flag
            detection_enabled: Whether to run detection
            scheduling_minute: Optional overridden minute for scheduling

        Returns:
            True if anomaly was detected, False otherwise
        """
        if not shutdown_flag.should_continue():
            return False

        current_minute = scheduling_minute if scheduling_minute is not None else current_time.minute

        # Step 1: Calculate metrics for all windows
        l1_metrics_by_window, l2_metrics_by_window = self._calculate_metrics_for_windows(
            entity_id, entity_df, current_time, current_minute, shutdown_flag
        )

        # Step 2: Run detection if enabled
        if not detection_enabled:
            return False

        if 60 not in l1_metrics_by_window and 60 not in l2_metrics_by_window:
            return False

        return self._run_detection_and_logging(
            entity_id, entity_df, l1_metrics_by_window, l2_metrics_by_window
        )

    def _calculate_metrics_for_windows(self, entity_id: str, entity_df: pd.DataFrame,
                                        current_time: datetime, current_minute: int,
                                        shutdown_flag) -> tuple:
        """
        Calculate L1 and L2 metrics for all observation windows.

        Args:
            entity_id: Entity identifier
            entity_df: DataFrame with entity logs
            current_time: Current processing time
            current_minute: Minute for scheduling decisions
            shutdown_flag: Graceful shutdown flag

        Returns:
            Tuple (l1_metrics_by_window, l2_metrics_by_window)
        """
        l1_metrics_by_window = {}
        l2_metrics_by_window = {}

        # Pre-filter data for each window
        window_data = {}
        for window in [60, 30, 10]:
            start_time, end_time = get_time_range(current_time, window)
            window_df = entity_df[(entity_df['@timestamp'] >= start_time) &
                                 (entity_df['@timestamp'] < end_time)]
            if not window_df.empty:
                window_data[window] = window_df

        # Calculate metrics for each window
        for window in [60, 30, 10]:
            if not shutdown_flag.should_continue():
                break

            window_df = window_data.get(window)
            if window_df is None:
                continue

            # L1 metrics
            try:
                l1_metrics = self.l1_calc.calculate(window_df, window)
                l1_metrics_by_window[window] = l1_metrics

                if self.scheduler.should_save_metrics('L1', window, current_minute):
                    self.storage.save_l1_metrics(entity_id, current_time, window, l1_metrics)
            except (ValueError, TypeError, KeyError) as e:
                self.logger.log_error(
                    f"L1 metrics calculation failed for entity {entity_id}, window {window}", e
                )
                continue

            # L2 metrics
            if self.l2_enabled:
                try:
                    l2_results = []
                    for dimension in self.l2_calc.dimensions:
                        dim_results = self.l2_calc.calculate(window_df, dimension)
                        l2_results.extend(dim_results)

                    l2_metrics_by_window[window] = l2_results

                    if self.scheduler.should_save_metrics('L2', window, current_minute):
                        self.storage.save_l2_metrics(entity_id, current_time, window, l2_results)
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.log_error(
                        f"L2 metrics calculation failed for entity {entity_id}, window {window}", e
                    )

        return l1_metrics_by_window, l2_metrics_by_window

    def _run_detection_and_logging(self, entity_id: str, entity_df: pd.DataFrame,
                                    l1_metrics_by_window: Dict, l2_metrics_by_window: Dict) -> bool:
        """
        Run anomaly detection and log results.

        Flow:
        1. Run anomaly detection
        2. Log alert immediately (with alert_id for correlation)
        3. For critical alerts, run LLM async and update OpenSearch document

        Args:
            entity_id: Entity identifier
            entity_df: DataFrame with entity logs
            l1_metrics_by_window: L1 metrics by window
            l2_metrics_by_window: L2 metrics by window

        Returns:
            True if anomaly detected, False otherwise
        """
        try:
            result = self.analyzer.analyze(self.detector, entity_id, l1_metrics_by_window, l2_metrics_by_window)

            if result is None:
                return False

            # Log alert IMMEDIATELY (don't wait for LLM)
            alert_id = self.logger.log_anomaly_alert(entity_id, result)

            # Run LLM analysis AFTER alert is sent (async approach)
            # LLM will update the OpenSearch document using alert_id
            if self.llm_analyzer and self.llm_analyzer.should_analyze(result):
                self._run_llm_analysis(entity_id, entity_df, result, l1_metrics_by_window, alert_id)

            return True

        except (RuntimeError, ValueError, KeyError) as e:
            self.logger.log_error(f"Anomaly detection failed for entity {entity_id}", e)
            return False

    def _run_llm_analysis(self, entity_id: str, entity_df: pd.DataFrame, anomaly_result: Dict,
                          l1_metrics_by_window: Dict, alert_id: str) -> None:
        """
        Run LLM analysis on anomaly and update OpenSearch document.

        Args:
            entity_id: Entity identifier
            entity_df: DataFrame with entity logs
            anomaly_result: Anomaly detection result
            l1_metrics_by_window: L1 metrics by window
            alert_id: Alert identifier for finding document in OpenSearch
        """
        try:
            # Use the configured trigger window for metrics
            trigger_window = self.llm_analyzer.trigger_window
            metrics = l1_metrics_by_window.get(
                trigger_window,
                l1_metrics_by_window.get(30, l1_metrics_by_window.get(60, {}))
            )

            llm_result = self.llm_analyzer.analyze(entity_id, anomaly_result, metrics, entity_df)

            if llm_result and not llm_result.get('error'):
                # Update the existing alert document in OpenSearch with LLM analysis
                self.logger.update_alert_with_llm(alert_id, llm_result)

        except Exception as e:
            self.logger.log_error(f"LLM analysis failed for entity {entity_id}", e)
