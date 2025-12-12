import numpy as np
import torch
from .model_loader import ModelLoader


FEATURE_ORDER = [
    'total_requests', 'mean_requests_per_minute', 'max_requests_per_minute',
    'std_requests_per_minute', 'cv_request_rate', 'peak_to_average_ratio', 'burst_score',
    'pct_2xx_responses', 'pct_3xx_responses', 'pct_4xx_responses', 'pct_5xx_responses',
    'error_rate', 'critical_error_rate', 'status_code_entropy', 'unique_status_codes',
    'mean_response_time', 'std_response_time', 'p50_response_time', 'p75_response_time',
    'p90_response_time', 'p95_response_time', 'p99_response_time',
    'pct_slow_requests', 'pct_very_slow_requests',
    'unique_source_ips', 'mean_requests_per_ip', 'max_requests_single_ip',
    'gini_ip_distribution', 'ip_concentration_top10pct',
    'unique_api_modules', 'module_entropy', 'top_module_percentage', 'module_switching_frequency',
    'unique_routes', 'route_entropy', 'top5_routes_percentage',
    'mean_response_size', 'std_response_size', 'max_response_size', 'min_response_size',
    'unique_user_agents', 'user_agent_entropy', 'bot_like_ua_percentage',
    'unique_http_methods', 'get_request_ratio', 'post_request_ratio'
]


class AnomalyDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.min_samples = config.get('detection', {}).get('min_samples_for_detection', 10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect(self, entity_id: str, metrics: dict, layer: str, **kwargs) -> tuple:
        metrics_vector = self._metrics_to_vector(metrics, layer)

        if metrics_vector is None:
            return False, 0.0, None, None

        if layer == 'L1':
            if self.model_loader.has_entity_model(entity_id):
                return self._detect_with_entity_model(entity_id, metrics_vector)
            else:
                return self._detect_with_cluster_model(entity_id, metrics_vector)
        elif layer == 'L2':
            user_id = kwargs.get('dimension_value')
            return self._detect_l2_simple(metrics_vector, user_id)
        else:
            return False, 0.0, None, None

    def _detect_with_entity_model(self, entity_id: str, metrics_vector: np.ndarray) -> tuple:
        model = self.model_loader.get_entity_model(entity_id)
        scaler = self.model_loader.get_entity_scaler(entity_id)
        threshold = self.model_loader.get_entity_threshold(entity_id)

        if model is None or scaler is None:
            return False, 0.0, None, None

        try:
            X_scaled = scaler.transform(metrics_vector)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            model.eval()
            with torch.no_grad():
                mse = model.reconstruction_error(X_tensor).cpu().item()

            is_anomaly = mse > threshold

            return is_anomaly, float(mse), f"entity_{entity_id}", None
        except (ValueError, TypeError, RuntimeError) as e:
            import sys
            print(f"WARNING: Entity model detection failed for {entity_id}: {str(e)}", file=sys.stderr)
            return False, 0.0, None, None

    def _detect_with_cluster_model(self, entity_id: str, metrics_vector: np.ndarray) -> tuple:
        cluster_id = self.model_loader.predict_cluster(metrics_vector)

        if cluster_id is None:
            return False, 0.0, None, None

        model = self.model_loader.get_cluster_model(cluster_id)
        scaler = self.model_loader.get_cluster_scaler(cluster_id)
        threshold = self.model_loader.get_cluster_threshold(cluster_id)

        if model is None or scaler is None:
            return False, 0.0, None, None

        try:
            X_scaled = scaler.transform(metrics_vector)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            model.eval()
            with torch.no_grad():
                mse = model.reconstruction_error(X_tensor).cpu().item()

            is_anomaly = mse > threshold

            return is_anomaly, float(mse), f"cluster_{cluster_id}", cluster_id
        except (ValueError, TypeError, RuntimeError) as e:
            import sys
            print(f"WARNING: Cluster model detection failed for entity {entity_id}, cluster {cluster_id}: {str(e)}", file=sys.stderr)
            return False, 0.0, None, None

    def _detect_l2_simple(self, metrics_vector: np.ndarray, user_id: str = None) -> tuple:
        # Default fallback if no model
        if user_id is None:
             error_rate = metrics_vector[0, 1] if metrics_vector.shape[1] > 1 else 0
             is_anomaly = error_rate > 50.0
             return is_anomaly, float(error_rate), "l2_simple_fallback", None

        model = self.model_loader.get_user_model(user_id)
        scaler = self.model_loader.get_user_scaler(user_id)

        if model is None or scaler is None:
             # Fallback to simple rule if no model yet
             error_rate = metrics_vector[0, 1] if metrics_vector.shape[1] > 1 else 0
             is_anomaly = error_rate > 50.0
             return is_anomaly, float(error_rate), "l2_simple_fallback", None

        try:
            X_scaled = scaler.transform(metrics_vector)
            
            prediction = model.predict(X_scaled)[0]
            score = model.decision_function(X_scaled)[0]
            
            is_anomaly = (prediction == -1)
            
            anomaly_score = -score 
            
            return is_anomaly, float(anomaly_score), f"user_{user_id}", None

        except (ValueError, TypeError, AttributeError) as e:
            import sys
            print(f"WARNING: User Isolation Forest detection failed for user {user_id}: {str(e)}", file=sys.stderr)
            return False, 0.0, None, None

    def _metrics_to_vector(self, metrics: dict, layer: str) -> np.ndarray:
        if layer == 'L1':
            vector = []
            for feature in FEATURE_ORDER:
                vector.append(metrics.get(feature, 0))
            return np.array(vector).reshape(1, -1)
        elif layer == 'L2':
            l2_features = ['request_count', 'error_rate', 'success_rate',
                          'mean_response_time', 'p95_response_time']
            vector = []
            for feature in l2_features:
                vector.append(metrics.get(feature, 0))
            return np.array(vector).reshape(1, -1)
        return None
