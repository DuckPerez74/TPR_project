from datetime import datetime


class HierarchicalAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.parallel_layers = config.get('detection', {}).get('parallel_layers', True)
        self.hierarchical_drill_down = config.get('detection', {}).get('hierarchical_drill_down', True)

    def analyze(self, detector, entity_id: str, l1_metrics_by_window: dict,
                l2_metrics_by_window: dict) -> dict:

        if not self.hierarchical_drill_down:
            return None

        results_60 = self._evaluate_window(detector, entity_id, 60,
                                           l1_metrics_by_window.get(60),
                                           l2_metrics_by_window.get(60))

        if not results_60['has_anomaly']:
            return None

        results_30 = self._evaluate_window(detector, entity_id, 30,
                                           l1_metrics_by_window.get(30),
                                           l2_metrics_by_window.get(30))

        results_10 = self._evaluate_window(detector, entity_id, 10,
                                           l1_metrics_by_window.get(10),
                                           l2_metrics_by_window.get(10))

        selected_window = self._select_smallest_anomalous_window(results_60, results_30, results_10)

        return {
            'entity_id': entity_id,
            'selected_window': selected_window,
            'results': {
                '60': results_60,
                '30': results_30,
                '10': results_10
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    def _evaluate_window(self, detector, entity_id: str, window: int,
                        l1_metrics: dict, l2_metrics: dict) -> dict:

        result = {
            'window': window,
            'has_anomaly': False,
            'anomaly_layer': None,
            'score': 0.0,
            'model_used': None,
            'cluster_id': None,
            'l2_dimension': None,
            'l2_dimension_value': None,
            'l2_details': []
        }

        if not l1_metrics and not l2_metrics:
            return result

        l1_anomaly = False
        l1_score = 0.0
        l1_model = None
        l1_cluster = None

        if l1_metrics:
            l1_anomaly, l1_score, l1_model, l1_cluster = detector.detect(entity_id, l1_metrics, 'L1')

        l2_anomaly = False
        l2_score = 0.0
        l2_dimension = None
        l2_dimension_value = None
        l2_details = []

        if self.parallel_layers and l2_metrics:
            for dim_result in l2_metrics:
                dim_metrics = dim_result.get('metrics', {})
                if dim_metrics:
                    dim_value = dim_result.get('dimension_value')
                    dimension = dim_result.get('dimension')
                    anomaly, score, model_used, _ = detector.detect(entity_id, dim_metrics, 'L2', dimension_value=dim_value, dimension=dimension)

                    if anomaly:
                        l2_details.append({
                            'dimension': dimension,
                            'dimension_value': dim_value,
                            'score': score,
                            'model_used': model_used
                        })

                        if score > l2_score:
                            l2_anomaly = True
                            l2_score = score
                            l2_dimension = dimension
                            l2_dimension_value = dim_value

        has_anomaly = l1_anomaly or (self.parallel_layers and l2_anomaly)

        if has_anomaly:
            if l1_score >= l2_score:
                result['anomaly_layer'] = 'L1'
                result['score'] = l1_score
                result['model_used'] = l1_model
                result['cluster_id'] = l1_cluster
            else:
                result['anomaly_layer'] = 'L2'
                result['score'] = l2_score
                result['l2_dimension'] = l2_dimension
                result['l2_dimension_value'] = l2_dimension_value
                result['l2_details'] = l2_details

            result['has_anomaly'] = True

        return result

    def _select_smallest_anomalous_window(self, r60: dict, r30: dict, r10: dict) -> int:
        if r10['has_anomaly']:
            return 10
        elif r30['has_anomaly']:
            return 30
        else:
            return 60
