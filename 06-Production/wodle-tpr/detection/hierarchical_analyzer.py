from datetime import datetime


class HierarchicalAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.parallel_layers = config.get('detection', {}).get('parallel_layers', True)
        self.hierarchical_drill_down = config.get('detection', {}).get('hierarchical_drill_down', True)

        # Risk scoring configuration
        risk_config = config.get('detection', {}).get('risk_scoring', {})
        self.use_risk_matrix = risk_config.get('enabled', False)
        self.risk_weights = risk_config.get('weights', {
            'l1': 0.3,
            'l2_user': 0.5,
            'l2_route': 0.2
        })
        self.risk_threshold = risk_config.get('threshold', 0.08)

        # Fallback: Minimum absolute score (if risk matrix disabled)
        self.min_score_for_alert = config.get('detection', {}).get('min_score_for_alert', 0.05)

    def analyze(self, detector, entity_id: str, l1_metrics_by_window: dict,
                l2_metrics_by_window: dict) -> dict:

        if not self.hierarchical_drill_down:
            return None

        results_60 = self._evaluate_window(detector, entity_id, 60,
                                           l1_metrics_by_window.get(60),
                                           l2_metrics_by_window.get(60))

        # Risk matrix enabled: Use composite scoring
        if self.use_risk_matrix:
            # Check if 60min window warrants drill-down based on risk score
            risk_score_60 = self._calculate_risk_score(results_60)

            if risk_score_60 < self.risk_threshold:
                # Risk too low, no alert
                return None
        else:
            # Fallback: Simple threshold filter
            if not results_60['has_anomaly'] or results_60['score'] < self.min_score_for_alert:
                return None

        # Drill down to smaller windows
        results_30 = self._evaluate_window(detector, entity_id, 30,
                                           l1_metrics_by_window.get(30),
                                           l2_metrics_by_window.get(30))

        results_10 = self._evaluate_window(detector, entity_id, 10,
                                           l1_metrics_by_window.get(10),
                                           l2_metrics_by_window.get(10))

        # Select smallest window based on risk matrix or simple anomaly
        if self.use_risk_matrix:
            selected_window, final_risk_score = self._select_window_by_risk(results_60, results_30, results_10)
        else:
            selected_window = self._select_smallest_anomalous_window(results_60, results_30, results_10)
            final_risk_score = None

        result = {
            'entity_id': entity_id,
            'selected_window': selected_window,
            'results': {
                '60': results_60,
                '30': results_30,
                '10': results_10
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        # Add risk score to result if enabled
        if self.use_risk_matrix and final_risk_score is not None:
            result['risk_score'] = final_risk_score
            result['risk_components'] = self._get_risk_components(
                result['results'][str(selected_window)]
            )

        return result

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

    def _calculate_risk_score(self, window_result: dict) -> float:
        """
        Calculate composite risk score from L1, L2 User, and L2 Route scores.

        Risk Score = (L1_score × w1) + (L2_User_score × w2) + (L2_Route_score × w3)

        Args:
            window_result: Result from _evaluate_window with L1 and L2 details

        Returns:
            Weighted risk score (0.0 - 1.0)
        """
        l1_score = 0.0
        l2_user_score = 0.0
        l2_route_score = 0.0

        # L1 score (if L1 anomaly detected)
        if window_result.get('anomaly_layer') == 'L1':
            l1_score = window_result.get('score', 0.0)

        # L2 scores from l2_details
        l2_details = window_result.get('l2_details', [])
        for detail in l2_details:
            dimension = detail.get('dimension')
            score = detail.get('score', 0.0)

            if dimension == 'user':
                # Take max score if multiple users (most severe)
                l2_user_score = max(l2_user_score, score)
            elif dimension == 'route':
                # Take max score if multiple routes
                l2_route_score = max(l2_route_score, score)

        # Calculate weighted risk score
        risk_score = (
            l1_score * self.risk_weights['l1'] +
            l2_user_score * self.risk_weights['l2_user'] +
            l2_route_score * self.risk_weights['l2_route']
        )

        return risk_score

    def _select_window_by_risk(self, r60: dict, r30: dict, r10: dict) -> tuple:
        """
        Select window with highest risk score.

        Args:
            r60, r30, r10: Window results

        Returns:
            Tuple (selected_window, risk_score)
        """
        risk_60 = self._calculate_risk_score(r60)
        risk_30 = self._calculate_risk_score(r30)
        risk_10 = self._calculate_risk_score(r10)

        # Select window with highest risk that exceeds threshold
        risks = [(10, risk_10), (30, risk_30), (60, risk_60)]
        risks_above_threshold = [(w, r) for w, r in risks if r >= self.risk_threshold]

        if risks_above_threshold:
            # Pick smallest window with risk >= threshold
            selected = min(risks_above_threshold, key=lambda x: x[0])
            return selected[0], selected[1]
        else:
            # Fallback: return window with max risk (even if below threshold)
            selected = max(risks, key=lambda x: x[1])
            return selected[0], selected[1]

    def _get_risk_components(self, window_result: dict) -> dict:
        """Get breakdown of risk score components for logging."""
        l1_score = 0.0
        l2_user_score = 0.0
        l2_route_score = 0.0

        if window_result.get('anomaly_layer') == 'L1':
            l1_score = window_result.get('score', 0.0)

        l2_details = window_result.get('l2_details', [])
        for detail in l2_details:
            dimension = detail.get('dimension')
            score = detail.get('score', 0.0)

            if dimension == 'user':
                l2_user_score = max(l2_user_score, score)
            elif dimension == 'route':
                l2_route_score = max(l2_route_score, score)

        return {
            'l1_score': round(l1_score, 6),
            'l2_user_score': round(l2_user_score, 6),
            'l2_route_score': round(l2_route_score, 6),
            'l1_weighted': round(l1_score * self.risk_weights['l1'], 6),
            'l2_user_weighted': round(l2_user_score * self.risk_weights['l2_user'], 6),
            'l2_route_weighted': round(l2_route_score * self.risk_weights['l2_route'], 6)
        }

    def _select_smallest_anomalous_window(self, r60: dict, r30: dict, r10: dict) -> int:
        if r10['has_anomaly']:
            return 10
        elif r30['has_anomaly']:
            return 30
        else:
            return 60
