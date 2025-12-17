import pandas as pd
from datetime import datetime
from .log_compressor import LogCompressor


class ContextBuilder:
    """
    Builds the complete analysis context package for LLM analysis.
    Combines alert metadata, anomaly details, metrics, and compressed logs.
    """

    def __init__(self):
        self.log_compressor = LogCompressor()

    def build_context(self, entity_id: str, anomaly_result: dict,
                      metrics: dict, entity_df: pd.DataFrame,
                      window_minutes: int = 10) -> dict:
        """
        Build complete context for LLM analysis.

        Args:
            entity_id: Entity identifier
            anomaly_result: Result from HierarchicalAnalyzer
            metrics: Current metrics snapshot
            entity_df: DataFrame with entity logs
            window_minutes: Time window in minutes

        Returns:
            Complete context dictionary for LLM

        Raises:
            ValueError: If context data is insufficient or invalid
        """
        # Validate inputs
        if entity_df is None or entity_df.empty:
            raise ValueError("entity_df is empty or None")

        if not anomaly_result or 'results' not in anomaly_result:
            raise ValueError("anomaly_result is missing or invalid")

        window_result = anomaly_result.get('results', {}).get(str(window_minutes), {})
        if not window_result:
            raise ValueError(f"No results found for window {window_minutes} minutes")

        compressed_logs = self.log_compressor.compress_logs(entity_df)
        if not compressed_logs:
            raise ValueError("No logs could be compressed from entity_df")

        logs_summary = self.log_compressor.get_logs_summary(compressed_logs)

        context = {
            'alert_context': {
                'entity_id': entity_id,
                'window_minutes': window_minutes,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            'anomaly_details': {
                'layer': window_result.get('anomaly_layer', 'L1'),
                'score': round(window_result.get('score', 0), 4),
                'model_used': window_result.get('model_used', 'unknown'),
                'l2_dimension': window_result.get('l2_dimension'),
                'l2_dimension_value': window_result.get('l2_dimension_value'),
                'has_anomaly': window_result.get('has_anomaly', False)
            },
            'logs_summary': logs_summary,
            'logs': compressed_logs
        }

        if metrics:
            context['metrics_snapshot'] = self._extract_key_metrics(metrics)

        return context

    def _extract_key_metrics(self, metrics: dict) -> dict:
        """Extract key metrics for context."""
        key_fields = [
            'error_rate', 'critical_error_rate', 'pct_4xx_responses', 'pct_5xx_responses',
            'total_requests', 'unique_source_ips', 'burst_score',
            'mean_response_time', 'p95_response_time',
            'unique_routes', 'route_entropy'
        ]

        extracted = {}
        for field in key_fields:
            if field in metrics:
                value = metrics[field]
                if isinstance(value, float):
                    extracted[field] = round(value, 4)
                else:
                    extracted[field] = value

        return extracted

    def format_for_prompt(self, context: dict) -> str:
        """
        Format context as readable text for LLM prompt.

        Args:
            context: Context dictionary from build_context

        Returns:
            Formatted string for prompt injection
        """
        lines = []

        alert = context.get('alert_context', {})
        lines.append(f"## Alert Information")
        lines.append(f"- Entity: {alert.get('entity_id')}")
        lines.append(f"- Window: {alert.get('window_minutes')} minutes")
        lines.append(f"- Timestamp: {alert.get('timestamp')}")
        lines.append("")

        anomaly = context.get('anomaly_details', {})
        lines.append(f"## Anomaly Detection")
        lines.append(f"- Layer: {anomaly.get('layer')}")
        lines.append(f"- Score: {anomaly.get('score')}")
        lines.append(f"- Model: {anomaly.get('model_used')}")
        if anomaly.get('l2_dimension'):
            lines.append(f"- Dimension: {anomaly.get('l2_dimension')} = {anomaly.get('l2_dimension_value')}")
        lines.append("")

        if context.get('metrics_snapshot'):
            lines.append("## Metrics Snapshot")
            for key, value in context['metrics_snapshot'].items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        summary = context.get('logs_summary', {})
        if summary:
            lines.append("## Logs Summary")
            lines.append(f"- Total logs: {summary.get('total_logs', 0)}")
            lines.append(f"- Methods: {summary.get('methods', {})}")
            lines.append(f"- Status groups: {summary.get('status_groups', {})}")
            lines.append(f"- Top routes: {summary.get('top_routes', {})}")
            lines.append("")

        logs = context.get('logs', [])
        if logs:
            lines.append("## Log Entries")
            lines.append("```json")
            import json
            lines.append(json.dumps(logs, indent=None, separators=(',', ':')))
            lines.append("```")

        return "\n".join(lines)
