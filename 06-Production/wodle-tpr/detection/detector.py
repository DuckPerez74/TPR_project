import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
from .model_loader import ModelLoader
from .window_buffer import WindowBuffer
from .model_assignment_cache import ModelAssignmentCache
from .voting_detector import VotingDetector
from constants import (
    L1_FEATURE_ORDER, L2_USER_FEATURES, L2_ROUTE_FEATURES,
    L1_SCORE_MULTIPLIER, L2_NORM_INPUT_MIN, L2_NORM_INPUT_MAX
)


class AnomalyDetector:
    def __init__(self, config: dict, opensearch_client=None):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.min_samples = config.get('detection', {}).get('min_samples_for_detection', 10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize WindowBuffer if cluster prediction is enabled
        cluster_config = config.get('detection', {}).get('cluster_prediction', {})
        if cluster_config.get('enabled', True):
            if opensearch_client is None:
                # Get singleton instance if not provided
                from core.opensearch_client import OpenSearchClient
                opensearch_client = OpenSearchClient.get_instance(config)

            self.window_buffer = WindowBuffer(opensearch_client, config)

            # Initialize Model Assignment Cache
            cache_path = Path(__file__).parent.parent / 'model_assignments.db'
            cluster_ttl = cluster_config.get('lookback_days', 7)
            self.assignment_cache = ModelAssignmentCache(cache_path, cluster_ttl_days=cluster_ttl)
        else:
            self.window_buffer = None
            self.assignment_cache = None
            opensearch_client = None

        # Use constants for score normalization (ML-tuned values)
        self.l1_norm_multiplier = L1_SCORE_MULTIPLIER
        self.l2_norm_min = L2_NORM_INPUT_MIN
        self.l2_norm_max = L2_NORM_INPUT_MAX

        # Initialize VotingDetector for L2 user dimension
        self.voting_detector = VotingDetector(
            self.model_loader, opensearch_client, self._normalize_l2_score
        )


    def _normalize_l1_score(self, mse: float, threshold: float) -> float:
        """Normalize L1 (autoencoder) MSE score to 0-1 range.
        
        Values at threshold = 0.5, values at 2x threshold = 1.0
        """
        if threshold <= 0:
            return min(1.0, mse)
        return min(1.0, mse / (threshold * self.l1_norm_multiplier))

    def _normalize_l2_score(self, score: float) -> float:
        """Normalize L2 (Isolation Forest) score to 0-1 range.
        
        Maps typical range [-0.3, 0.5] to [0, 1]
        """
        range_size = self.l2_norm_max - self.l2_norm_min
        if range_size <= 0:
            return min(1.0, max(0.0, score))
        return min(1.0, max(0.0, (score - self.l2_norm_min) / range_size))

    def detect(self, entity_id: str, metrics: dict, layer: str, **kwargs) -> tuple[bool, float, str, Optional[int], Optional[dict]]:
        """Detect anomalies. Returns (is_anomaly, score, model_used, cluster_id, voting_details)."""
        dimension = kwargs.get('dimension')
        metrics_vector = self._metrics_to_vector(metrics, layer, dimension)

        if metrics_vector is None:
            return False, 0.0, "invalid_metrics_vector", None, None

        if layer == 'L1':
            if self.model_loader.has_entity_model(entity_id):
                result = self._detect_with_entity_model(entity_id, metrics_vector)
                return result + (None,)  # Add None for voting_details
            else:
                result = self._detect_with_cluster_model(entity_id, metrics_vector)
                return result + (None,)  # Add None for voting_details
        elif layer == 'L2':
            dimension_value = kwargs.get('dimension_value')
            return self._detect_l2_simple(entity_id, metrics_vector, metrics, dimension_value, dimension)
        else:
            return False, 0.0, "unknown_layer", None, None

    def _detect_with_entity_model(self, entity_id: str, metrics_vector: np.ndarray) -> tuple:
        model = self.model_loader.get_entity_model(entity_id)
        scaler = self.model_loader.get_entity_scaler(entity_id)
        threshold = self.model_loader.get_entity_threshold(entity_id)

        if model is None or scaler is None:
            return False, 0.0, "entity_model_not_found", None

        try:
            X_scaled = scaler.transform(metrics_vector)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            model.eval()
            with torch.no_grad():
                mse = model.reconstruction_error(X_tensor).cpu().item()

            is_anomaly = mse > threshold
            normalized_score = self._normalize_l1_score(mse, threshold)

            return is_anomaly, float(normalized_score), f"entity_{entity_id}", None
        except (ValueError, TypeError, RuntimeError) as e:
            import sys
            print(f"WARNING: Entity model detection failed for {entity_id}: {str(e)}", file=sys.stderr)
            return False, 0.0, "entity_model_error", None

    def _detect_with_cluster_model(self, entity_id: str, metrics_vector: np.ndarray) -> tuple:
        # Check assignment cache FIRST (optimization)
        if self.assignment_cache is not None:
            assignment = self.assignment_cache.get(entity_id)

            if assignment:
                # Cache HIT - use assigned model directly (FAST PATH)
                if assignment.model_type == 'entity':
                    # Entity now has its own model (was retrained)
                    return self._detect_with_entity_model(entity_id, metrics_vector)
                else:
                    # Use cached cluster assignment
                    cluster_id = assignment.cluster_id
            else:
                # Cache MISS - need to determine cluster (SLOW PATH, only first time)
                cluster_id = self._determine_and_cache_cluster(entity_id, metrics_vector)
        else:
            # Assignment cache disabled - use old behavior
            cluster_id = self._determine_cluster(entity_id, metrics_vector)

        if cluster_id is None:
            return False, 0.0, "cluster_not_determined", None

        model = self.model_loader.get_cluster_model(cluster_id)
        scaler = self.model_loader.get_cluster_scaler(cluster_id)
        threshold = self.model_loader.get_cluster_threshold(cluster_id)

        if model is None or scaler is None:
            return False, 0.0, "cluster_model_not_found", None

        try:
            X_scaled = scaler.transform(metrics_vector)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            model.eval()
            with torch.no_grad():
                mse = model.reconstruction_error(X_tensor).cpu().item()

            is_anomaly = mse > threshold
            normalized_score = self._normalize_l1_score(mse, threshold)

            return is_anomaly, float(normalized_score), f"cluster_{cluster_id}", cluster_id
        except (ValueError, TypeError, RuntimeError) as e:
            import sys
            print(f"WARNING: Cluster model detection failed for entity {entity_id}, cluster {cluster_id}: {str(e)}", file=sys.stderr)
            return False, 0.0, "cluster_model_error", None

    def _determine_and_cache_cluster(self, entity_id: str, metrics_vector: np.ndarray) -> int:
        """
        Determine cluster for entity and cache the assignment.
        Used when assignment cache is enabled but entity not in cache.
        """
        import sys

        # First check if entity NOW has its own model (might have been trained recently)
        if self.model_loader.has_entity_model(entity_id):
            # Register and use entity model - verify it exists
            model_exists = self.model_loader.get_entity_model(entity_id) is not None
            self.assignment_cache.set_entity_model(entity_id, f"entity_{entity_id}", model_exists)
            print(f"[Detector] Entity {entity_id} now has own model, cached",
                  file=sys.stderr)
            return None  # Will be handled by caller switching to entity model

        # Determine cluster using WindowBuffer
        cluster_id = self._determine_cluster(entity_id, metrics_vector)

        if cluster_id is not None:
            # Try to get confidence if possible
            confidence = None
            if self.window_buffer is not None:
                recent_windows = self.window_buffer.get_recent_windows(entity_id)
                if recent_windows is not None and len(recent_windows) > 0:
                    result = self.model_loader.predict_cluster_with_confidence(
                        np.mean(recent_windows, axis=0)
                    )
                    if result:
                        confidence = result.get('confidence')

            # Cache the cluster assignment - verify model exists
            model_exists = self.model_loader.get_cluster_model(cluster_id) is not None
            self.assignment_cache.set_cluster_model(entity_id, cluster_id, confidence, model_exists)

        return cluster_id

    def _determine_cluster(self, entity_id: str, metrics_vector: np.ndarray) -> int:
        """
        Determine cluster for entity using WindowBuffer (if available) or single window.
        Used when assignment cache is disabled.
        """
        import sys

        if self.window_buffer is not None:
            recent_windows = self.window_buffer.get_recent_windows(entity_id, observation_window=60)

            if recent_windows is not None and len(recent_windows) > 0:
                # Predict cluster using aggregated windows (consistent with training)
                cluster_id = self.model_loader.predict_cluster_from_windows(recent_windows)
            else:
                # Fallback to single window prediction if insufficient historical data
                print(f"[Detector] Using single-window prediction for {entity_id} "
                      f"(insufficient historical data)", file=sys.stderr)
                cluster_id = self.model_loader.predict_cluster(metrics_vector)
        else:
            # WindowBuffer disabled, use single window (legacy behavior)
            cluster_id = self.model_loader.predict_cluster(metrics_vector)

        return cluster_id

    def _detect_l2_simple(self, entity_id: str, metrics_vector: np.ndarray,
                          metrics: dict, dimension_value: str = None, dimension: str = None) -> tuple:
        # Handle route dimension - skip if no model exists
        if dimension == 'route':
            if dimension_value is None:
                return False, 0.0, "l2_route_no_value", None, None

            # For routes, we only use route-specific models if they exist
            # If no model exists for this route, skip L2 route detection
            route_model = self.model_loader.get_route_model(dimension_value)
            if route_model is None:
                # Skip L2 route detection - no model available
                return False, 0.0, "l2_route_no_model_skip", None, None

            # Route has model, proceed with detection
            scaler = self.model_loader.get_route_scaler(dimension_value)
            result = self._run_isolation_forest_detection(metrics_vector, route_model, scaler, f"route_{dimension_value}")
            return result + (None,)  # Add None for voting_details

        # Handle user dimension
        elif dimension == 'user':
            if dimension_value is None:
                return False, 0.0, "l2_user_no_value", None, None

            user_id = dimension_value
            model = self.model_loader.get_user_model(user_id)
            scaler = self.model_loader.get_user_scaler(user_id)

            # If user has own model, use it
            if model is not None and scaler is not None:
                result = self._run_isolation_forest_detection(metrics_vector, model, scaler, f"user_{user_id}")
                return result + (None,)  # Add None for voting_details

            # User has no model - try voting with similar users
            account = metrics.get('account')
            if account and account != 'unknown':
                # Voting detector returns 5-tuple including voting_details
                return self.voting_detector.detect_with_voting(entity_id, user_id, account, metrics_vector)

            # No account type available - skip L2 detection
            return False, 0.0, "l2_user_no_account", None, None

        # Other dimensions or fallback
        else:
            error_rate = metrics_vector[0, 1] if metrics_vector.shape[1] > 1 else 0
            is_anomaly = error_rate > 50.0
            return is_anomaly, float(error_rate), "l2_simple_fallback", None, None

    def _run_isolation_forest_detection(self, metrics_vector: np.ndarray,
                                        model, scaler, identifier: str) -> tuple:
        """Run Isolation Forest detection with given model and scaler. Returns 4-tuple."""
        try:
            X_scaled = scaler.transform(metrics_vector)

            prediction = model.predict(X_scaled)[0]
            score = model.decision_function(X_scaled)[0]

            is_anomaly = (prediction == -1)
            anomaly_score = -score
            normalized_score = self._normalize_l2_score(anomaly_score)

            return is_anomaly, float(normalized_score), identifier, None

        except (ValueError, TypeError, AttributeError) as e:
            import sys
            print(f"WARNING: Isolation Forest detection failed for {identifier}: {str(e)}", file=sys.stderr)
            return False, 0.0, "isolation_forest_error", None


    def _metrics_to_vector(self, metrics: dict, layer: str, dimension: str = None) -> np.ndarray:
        if layer == 'L1':
            vector = []
            for feature in L1_FEATURE_ORDER:
                vector.append(metrics.get(feature, 0))
            return np.array(vector).reshape(1, -1)
        elif layer == 'L2':
            # Use appropriate feature order based on dimension
            if dimension == 'user':
                feature_order = L2_USER_FEATURES
            elif dimension == 'route':
                feature_order = L2_ROUTE_FEATURES
            else:
                # Fallback to user features
                feature_order = L2_USER_FEATURES

            vector = []
            for feature in feature_order:
                vector.append(metrics.get(feature, 0))
            return np.array(vector).reshape(1, -1)
        return None
