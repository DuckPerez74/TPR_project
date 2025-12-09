import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from config import ENTITY_MODEL_PATH, CLUSTER_MODEL_PATH, AUTOENCODER_PATH, OBSERVATION_WINDOWS, THRESHOLD_TYPE

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

class AnomalyDetector:
    def __init__(self):
        self.autoencoder = None
        self.entity_models = {}
        self.cluster_models = {}
        self.entity_thresholds = {}
        self.cluster_thresholds = {}
        self.load_models()
        self.load_thresholds()

    def load_models(self):
        if KERAS_AVAILABLE and AUTOENCODER_PATH.exists():
            try:
                self.autoencoder = keras.models.load_model(AUTOENCODER_PATH)
            except Exception:
                pass

        if ENTITY_MODEL_PATH.exists():
            for model_file in ENTITY_MODEL_PATH.glob("entity_*.pkl"):
                try:
                    entity_id = model_file.stem.replace("entity_", "")
                    with open(model_file, 'rb') as f:
                        self.entity_models[entity_id] = pickle.load(f)
                except Exception:
                    continue

        if CLUSTER_MODEL_PATH.exists():
            for model_file in CLUSTER_MODEL_PATH.glob("cluster_*.pkl"):
                try:
                    cluster_id = int(model_file.stem.split('_')[1])
                    with open(model_file, 'rb') as f:
                        self.cluster_models[cluster_id] = pickle.load(f)
                except Exception:
                    continue

    def load_thresholds(self):
        if ENTITY_MODEL_PATH.exists():
            for threshold_file in ENTITY_MODEL_PATH.glob("entity_*_thresholds.json"):
                try:
                    entity_id = threshold_file.stem.replace("entity_", "").replace("_thresholds", "")
                    with open(threshold_file, 'r') as f:
                        data = json.load(f)
                        self.entity_thresholds[entity_id] = data.get(THRESHOLD_TYPE, data.get('p95', 0.01))
                except Exception:
                    continue

        if CLUSTER_MODEL_PATH.exists():
            for threshold_file in CLUSTER_MODEL_PATH.glob("cluster_*_thresholds.json"):
                try:
                    cluster_id = int(threshold_file.stem.split('_')[1])
                    with open(threshold_file, 'r') as f:
                        data = json.load(f)
                        self.cluster_thresholds[cluster_id] = data.get(THRESHOLD_TYPE, data.get('p95', 0.01))
                except Exception:
                    continue

    def predict_cluster(self, metrics_vector):
        if self.autoencoder is None:
            return None

        try:
            encoded = self.autoencoder.predict(metrics_vector, verbose=0)
            cluster_id = np.argmax(encoded[0])
            return cluster_id
        except Exception:
            return None

    def detect_anomaly(self, entity_id, metrics_dict):
        metrics_vector = self._dict_to_vector(metrics_dict)

        if str(entity_id) in self.entity_models:
            return self._detect_with_entity_model(entity_id, metrics_vector)
        else:
            return self._detect_with_cluster_model(metrics_vector)

    def _detect_with_entity_model(self, entity_id, metrics_vector):
        entity_id_str = str(entity_id)
        model = self.entity_models[entity_id_str]
        threshold = self.entity_thresholds.get(entity_id_str, 0.01)

        try:
            reconstruction = model.predict(metrics_vector, verbose=0)
            mse = np.mean(np.square(metrics_vector - reconstruction))
            is_anomaly = mse > threshold

            return is_anomaly, float(mse), f"entity_{entity_id}", None
        except Exception:
            return False, 0.0, None, None

    def _detect_with_cluster_model(self, metrics_vector):
        cluster_id = self.predict_cluster(metrics_vector)

        if cluster_id is None or cluster_id not in self.cluster_models:
            return False, 0.0, None, None

        model = self.cluster_models[cluster_id]
        threshold = self.cluster_thresholds.get(cluster_id, 0.01)

        try:
            reconstruction = model.predict(metrics_vector, verbose=0)
            mse = np.mean(np.square(metrics_vector - reconstruction))
            is_anomaly = mse > threshold

            return is_anomaly, float(mse), f"cluster_{cluster_id}", cluster_id
        except Exception:
            return False, 0.0, None, None

    def _dict_to_vector(self, metrics_dict):
        feature_order = [
            'total_requests', 'mean_requests_per_minute', 'max_requests_per_minute',
            'min_requests_per_minute', 'std_requests_per_minute', 'cv_request_rate',
            'peak_to_average_ratio', 'burst_score', 'pct_2xx_responses', 'pct_3xx_responses',
            'pct_4xx_responses', 'pct_5xx_responses', 'error_rate', 'critical_error_rate',
            'status_code_entropy', 'unique_status_codes', 'mean_response_time',
            'std_response_time', 'p50_response_time', 'p75_response_time',
            'p90_response_time', 'p95_response_time', 'p99_response_time',
            'pct_slow_requests', 'pct_very_slow_requests', 'unique_source_ips',
            'mean_requests_per_ip', 'max_requests_single_ip', 'gini_ip_distribution',
            'ip_concentration_top10pct', 'unique_operators', 'unique_accounts',
            'account_diversity_ratio', 'unique_api_modules', 'module_entropy',
            'top_module_percentage', 'module_switching_frequency', 'unique_routes',
            'route_entropy', 'top5_routes_percentage', 'mean_response_size',
            'std_response_size', 'max_response_size', 'min_response_size',
            'unique_user_agents', 'user_agent_entropy', 'bot_like_ua_percentage',
            'unique_http_methods', 'get_request_ratio', 'post_request_ratio', 'http11_ratio'
        ]

        vector = []
        for feature in feature_order:
            vector.append(metrics_dict.get(feature, 0))

        return np.array(vector).reshape(1, -1)

def analyze_entity_windows(detector, entity_id, metrics_60, metrics_30, metrics_10):
    results = {}

    is_anomaly_60, score_60, model_used_60, cluster_60 = detector.detect_anomaly(entity_id, metrics_60)
    results['60'] = {'is_anomaly': is_anomaly_60, 'score': score_60, 'model': model_used_60, 'cluster': cluster_60}

    if not is_anomaly_60:
        return None

    is_anomaly_30, score_30, model_used_30, cluster_30 = detector.detect_anomaly(entity_id, metrics_30)
    results['30'] = {'is_anomaly': is_anomaly_30, 'score': score_30, 'model': model_used_30, 'cluster': cluster_30}

    is_anomaly_10, score_10, model_used_10, cluster_10 = detector.detect_anomaly(entity_id, metrics_10)
    results['10'] = {'is_anomaly': is_anomaly_10, 'score': score_10, 'model': model_used_10, 'cluster': cluster_10}

    if is_anomaly_10:
        selected_window = 10
    elif is_anomaly_30:
        selected_window = 30
    else:
        selected_window = 60

    return {
        'entity_id': entity_id,
        'selected_window': selected_window,
        'model_used': results[str(selected_window)]['model'],
        'cluster_id': results[str(selected_window)]['cluster'],
        'results': results,
        'timestamp': datetime.utcnow().isoformat()
    }
