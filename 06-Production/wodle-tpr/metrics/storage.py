from datetime import datetime, timedelta
from typing import List, Dict


class MetricsStorage:
    def __init__(self, client, config: dict):
        self.client = client
        self.base_index_name = config.get('indices', {}).get('metrics', {}).get('name', 'metrics-tpr')
        self.create_if_missing = config.get('indices', {}).get('metrics', {}).get('create_if_missing', True)
        self.buffer = []
        self.max_buffer_size = config.get('performance', {}).get('metrics_buffer_size', 1000)

    def _get_index_name(self, timestamp: datetime) -> str:
        """
        Generate monthly index name based on metric timestamp.

        Args:
            timestamp: The timestamp of the metric (from the log data)

        Returns:
            Index name like 'metrics-tpr-2025-11'
        """
        month_str = timestamp.strftime('%Y-%m')
        return f"{self.base_index_name}-{month_str}"

    def _ensure_index_exists(self, index_name: str):
        """Ensure the specified index exists, create if missing."""
        try:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to create index {index_name}: {str(e)}", file=sys.stderr)

    def get_index_pattern(self, timestamp: datetime = None, months_back: int = 0) -> str:
        """
        Get index pattern for querying multiple months.

        Args:
            timestamp: Reference timestamp (default: now)
            months_back: Number of months to include (0 = specific month only, >0 = use wildcard)

        Returns:
            Index pattern like 'metrics-tpr-2025-01' or 'metrics-tpr-*' for multiple months
        """
        if months_back == 0:
            if timestamp is None:
                timestamp = datetime.now()
            return self._get_index_name(timestamp)
        else:
            # For multiple months, use wildcard pattern
            return f"{self.base_index_name}-*"

    def flush_metrics(self):
        """Force flush current buffer to OpenSearch."""
        if not self.buffer:
            return

        self.save_metrics_bulk(self.buffer)
        self.buffer = []

    def save_l1_metrics(self, entity_id: str, timestamp: datetime, window: int, metrics: dict):
        start_time = timestamp - timedelta(minutes=window)

        document = {
            '@timestamp': timestamp.isoformat(),
            'window_start_time': start_time.isoformat(),
            'window_end_time': timestamp.isoformat(),
            'entity_id': entity_id,
            'observation_window': window,
            'layer': 'L1',
            'metric_type': 'entity_metric',
            'sample_count': metrics.get('total_requests', 0),
            'metrics': metrics
        }
        
        self.buffer.append(document)
        if len(self.buffer) >= self.max_buffer_size:
            self.flush_metrics()

    def save_l2_metrics(self, entity_id: str, timestamp: datetime, window: int, l2_results: List[Dict]):
        start_time = timestamp - timedelta(minutes=window)

        for result in l2_results:
            document = {
                '@timestamp': timestamp.isoformat(),
                'window_start_time': start_time.isoformat(),
                'window_end_time': timestamp.isoformat(),
                'entity_id': entity_id,
                'observation_window': window,
                'layer': 'L2',
                'metric_type': 'user_metric' if result['dimension'] == 'user' else 'l2_metric',
                'dimension': result['dimension'],
                'dimension_value': result['dimension_value'],
                'sample_count': result['sample_count'],
                'metrics': result['metrics']
            }
            if result['dimension'] == 'user':
                document['operator_id'] = result['dimension_value']
            
            self.buffer.append(document)

        if len(self.buffer) >= self.max_buffer_size:
            self.flush_metrics()


    def save_metrics_bulk(self, metrics_list: List[Dict]):
        """
        Save a large list of metrics using OpenSearch bulk API.
        Generates deterministic IDs for idempotency.
        Groups metrics by month to route to correct indices.
        """
        if not metrics_list:
            return

        from opensearchpy import helpers
        import hashlib

        # Group metrics by month to ensure they go to the correct index
        metrics_by_month = {}
        for metric in metrics_list:
            timestamp_str = metric.get('@timestamp')
            if not timestamp_str:
                continue

            # Parse timestamp to determine index
            if isinstance(timestamp_str, str):
                metric_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                metric_timestamp = timestamp_str

            index_name = self._get_index_name(metric_timestamp)

            if index_name not in metrics_by_month:
                metrics_by_month[index_name] = []

            metrics_by_month[index_name].append(metric)

        # Process each month's metrics
        for index_name, month_metrics in metrics_by_month.items():
            if self.create_if_missing:
                self._ensure_index_exists(index_name)

            actions = []
            for metric in month_metrics:
                # Generate deterministic ID
                # ID format: {entity_id}_{timestamp}_{window}_{layer}_{metric_type}_{dimension}_{value}
                unique_str = f"{metric['entity_id']}_{metric['@timestamp']}_{metric['observation_window']}_{metric['layer']}_{metric['metric_type']}"

                if 'dimension' in metric:
                    unique_str += f"_{metric['dimension']}"
                if 'dimension_value' in metric:
                    unique_str += f"_{metric['dimension_value']}"

                doc_id = hashlib.md5(unique_str.encode()).hexdigest()

                action = {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": metric
                }
                actions.append(action)

            try:
                success, failed = helpers.bulk(self.client, actions, stats_only=True)
                # print(f"  Bulk save to {index_name}: {success} loaded, {failed} failed")
            except Exception as e:
                import sys
                print(f"ERROR: Bulk save to {index_name} failed: {str(e)}", file=sys.stderr)
            finally:
                # Free actions list to prevent memory accumulation
                del actions

