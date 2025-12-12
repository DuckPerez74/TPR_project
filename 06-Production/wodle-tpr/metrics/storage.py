from datetime import datetime, timedelta
from typing import List, Dict


class MetricsStorage:
    def __init__(self, client, config: dict):
        self.client = client
        self.index_name = config.get('indices', {}).get('metrics', {}).get('name', 'metrics-tpr')
        self.create_if_missing = config.get('indices', {}).get('metrics', {}).get('create_if_missing', True)

        if self.create_if_missing:
            self._ensure_index_exists()

    def _ensure_index_exists(self):
        try:
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to create index {self.index_name}: {str(e)}", file=sys.stderr)

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

        try:
            self.client.index(index=self.index_name, body=document)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to save L1 metrics for entity {entity_id}: {str(e)}", file=sys.stderr)

    def save_l2_metrics(self, entity_id: str, timestamp: datetime, window: int, l2_results: List[Dict]):
        documents = []
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
            documents.append(document)

        if not documents:
            return

        try:
            bulk_body = []
            for doc in documents:
                bulk_body.append({'index': {'_index': self.index_name}})
                bulk_body.append(doc)

            if bulk_body:
                self.client.bulk(body=bulk_body)
        except Exception as e:
            import sys
            print(f"ERROR: Failed to save L2 metrics for entity {entity_id} ({len(documents)} docs): {str(e)}", file=sys.stderr)

    def save_metrics_bulk(self, metrics_list: List[Dict]):
        """
        Save a large list of metrics using OpenSearch bulk API.
        Generates deterministic IDs for idempotency.
        """
        if not metrics_list:
            return

        from opensearchpy import helpers
        import hashlib

        actions = []
        for metric in metrics_list:
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
                "_index": self.index_name,
                "_id": doc_id,
                "_source": metric
            }
            actions.append(action)

        try:
            success, failed = helpers.bulk(self.client, actions, stats_only=True)
            # print(f"  Bulk save: {success} loaded, {failed} failed")
        except Exception as e:
            import sys
            print(f"ERROR: Bulk save failed: {str(e)}", file=sys.stderr)

