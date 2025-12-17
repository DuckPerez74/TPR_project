import pickle
import json
import torch
from pathlib import Path
import joblib
from training.autoencoder import AutoEncoder
from constants import THRESHOLD_PERCENTILE, THRESHOLD_FALLBACK


class ModelLoader:
    def __init__(self, config: dict):
        models_config = config.get('models', {})
        base_path = Path(__file__).parent.parent
        observation_window = models_config.get('observation_window', 60)

        self.entity_models_path = base_path / models_config.get('entity_models_path', 'models/entity_models')
        self.cluster_models_path = base_path / models_config.get('cluster_models_path', 'models/cluster_models')
        self.user_models_path = base_path / models_config.get('user_models_path', 'models/user_models')
        self.route_models_path = base_path / models_config.get('route_models_path', 'models/route_models')
        self.kmeans_path = base_path / 'models' / f'kmeans_{observation_window}min.pkl'

        # Use constants for ML-tuned values
        self.threshold_fallback = THRESHOLD_FALLBACK
        self.threshold_percentile = THRESHOLD_PERCENTILE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Lazy loading caches
        self.entity_models = {}
        self.cluster_models = {}
        self.user_models = {}
        self.entity_scalers = {}
        self.cluster_scalers = {}
        self.user_scalers = {}
        self.entity_thresholds = {}
        self.cluster_thresholds = {}
        self.kmeans_clusterer = None
        
        # Load K-Means immediately as it's global and small
        self._load_kmeans()

    def _load_kmeans(self):
        if not self.kmeans_path.exists():
            return

        try:
            from training.kmeans_clusterer import KMeansClusterer
            self.kmeans_clusterer = KMeansClusterer()
            self.kmeans_clusterer.load(self.kmeans_path)
        except (FileNotFoundError, ImportError, pickle.UnpicklingError, Exception) as e:
            import sys
            print(f"WARNING: Failed to load K-means clusterer: {str(e)}", file=sys.stderr)
            print(f"  If you recently updated the code, you may need to retrain the K-means model", file=sys.stderr)

    def _load_entity_model_from_disk(self, entity_id: str):
        """Lazy load entity model from disk"""
        metrics_file = self.entity_models_path / f"entity_{entity_id}.pt"
        if not metrics_file.exists():
            return False

        try:
            checkpoint = torch.load(metrics_file, map_location=self.device)
            model = AutoEncoder(
                checkpoint['input_dim'],
                checkpoint['encoding_dim'],
                checkpoint['hidden_dim']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.entity_models[entity_id] = model

            scaler_file = self.entity_models_path / f"entity_{entity_id}_scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.entity_scalers[entity_id] = pickle.load(f)

            threshold_file = self.entity_models_path / f"entity_{entity_id}_thresholds.json"
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    thresholds = json.load(f)
                    self.entity_thresholds[entity_id] = thresholds.get(self.threshold_percentile, self.threshold_fallback)
            
            return True
        except Exception as e:
            import sys
            print(f"WARNING: Failed to load entity model {entity_id}: {str(e)}", file=sys.stderr)
            return False


    def _load_cluster_model_from_disk(self, cluster_id: int):
        """Lazy load cluster model from disk"""
        model_file = self.cluster_models_path / f"cluster_{cluster_id}.pt"
        if not model_file.exists():
            return False

        try:
            checkpoint = torch.load(model_file, map_location=self.device)
            model = AutoEncoder(
                checkpoint['input_dim'],
                checkpoint['encoding_dim'],
                checkpoint['hidden_dim']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.cluster_models[cluster_id] = model

            scaler_file = self.cluster_models_path / f"cluster_{cluster_id}_scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.cluster_scalers[cluster_id] = pickle.load(f)

            threshold_file = self.cluster_models_path / f"cluster_{cluster_id}_thresholds.json"
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    thresholds = json.load(f)
                    self.cluster_thresholds[cluster_id] = thresholds.get(self.threshold_percentile, self.threshold_fallback)
            
            return True
        except Exception as e:
            import sys
            print(f"WARNING: Failed to load cluster model {cluster_id}: {str(e)}", file=sys.stderr)
            return False

    def _load_user_model_from_disk(self, user_id_safe: str):
        """Lazy load user model from disk"""
        model_file = self.user_models_path / f"user_{user_id_safe}.joblib"
        if not model_file.exists():
            return False

        try:
            model = joblib.load(model_file)
            self.user_models[user_id_safe] = model

            scaler_file = self.user_models_path / f"user_{user_id_safe}_scaler.joblib"
            if scaler_file.exists():
                self.user_scalers[user_id_safe] = joblib.load(scaler_file)
            
            return True
        except Exception as e:
            import sys
            print(f"WARNING: Failed to load user model {user_id_safe}: {str(e)}", file=sys.stderr)
            return False

    def get_entity_model(self, entity_id: str):
        if entity_id not in self.entity_models:
            self._load_entity_model_from_disk(entity_id)
        return self.entity_models.get(entity_id)

    def get_entity_scaler(self, entity_id: str):
        if entity_id not in self.entity_scalers:
            self._load_entity_model_from_disk(entity_id)
        return self.entity_scalers.get(entity_id)


    def get_cluster_model(self, cluster_id: int):
        if cluster_id not in self.cluster_models:
            self._load_cluster_model_from_disk(cluster_id)
        return self.cluster_models.get(cluster_id)

    def get_cluster_scaler(self, cluster_id: int):
        if cluster_id not in self.cluster_scalers:
            self._load_cluster_model_from_disk(cluster_id)
        return self.cluster_scalers.get(cluster_id)

    def get_entity_threshold(self, entity_id: str) -> float:
        if entity_id not in self.entity_thresholds:
            self._load_entity_model_from_disk(entity_id)
        return self.entity_thresholds.get(entity_id, self.threshold_fallback)

    def get_cluster_threshold(self, cluster_id: int) -> float:
        if cluster_id not in self.cluster_thresholds:
            self._load_cluster_model_from_disk(cluster_id)
        return self.cluster_thresholds.get(cluster_id, self.threshold_fallback)

    def has_entity_model(self, entity_id: str) -> bool:
        # Check cache first
        if entity_id in self.entity_models:
            return True
        # Check disk without loading
        return (self.entity_models_path / f"entity_{entity_id}.pt").exists()

    def get_user_model(self, user_id: str):
        # We need to sanitize the input user_id to match the file storage convention
        safe_user_id = "".join([c if c.isalnum() else "_" for c in user_id])
        if safe_user_id not in self.user_models:
            self._load_user_model_from_disk(safe_user_id)
        return self.user_models.get(safe_user_id)

    def get_user_scaler(self, user_id: str):
        safe_user_id = "".join([c if c.isalnum() else "_" for c in user_id])
        if safe_user_id not in self.user_scalers:
            self._load_user_model_from_disk(safe_user_id)
        return self.user_scalers.get(safe_user_id)

    def get_available_clusters(self) -> list:
        # Scan directory for active clusters
        if not self.cluster_models_path.exists():
            return []
        
        clusters = []
        for f in self.cluster_models_path.glob("cluster_*.pt"):
            try:
                clusters.append(int(f.stem.split('_')[1]))
            except:
                pass
        return sorted(list(set(clusters)))


    def predict_cluster(self, metrics_vector):
        """
        Predict cluster from a single metrics vector.

        WARNING: This method uses a single observation window, which is inconsistent
        with how the model was trained (using mean of multiple windows).
        For better accuracy, consider using predict_cluster_from_windows() when possible.

        Args:
            metrics_vector: np.array of shape (1, n_features) or (n_features,)

        Returns:
            cluster_id: int or None
        """
        if self.kmeans_clusterer is None:
            clusters = self.get_available_clusters()
            return clusters[0] if clusters else None

        return self.kmeans_clusterer.predict(metrics_vector)

    def predict_cluster_from_windows(self, metrics_windows):
        """
        Predict cluster by aggregating multiple observation windows.

        This method maintains consistency with training by aggregating windows
        using mean before prediction. Use this method when you have access to
        multiple observation windows for an entity.

        Args:
            metrics_windows: np.array of shape (n_windows, n_features)

        Returns:
            cluster_id: int or None
        """
        if self.kmeans_clusterer is None:
            clusters = self.get_available_clusters()
            return clusters[0] if clusters else None

        return self.kmeans_clusterer.predict_from_windows(metrics_windows)

    def predict_cluster_with_confidence(self, metrics_vector):
        """
        Predict cluster and return confidence metrics.

        Args:
            metrics_vector: np.array of shape (1, n_features) or (n_features,)

        Returns:
            dict with cluster prediction and confidence metrics, or None
        """
        if self.kmeans_clusterer is None:
            return None

        return self.kmeans_clusterer.predict_with_confidence(metrics_vector)

    def has_any_models_on_disk(self) -> bool:
        """Check if any models exist on disk to determine if detection can run."""
        # Check for entity models
        if self.entity_models_path.exists():
            if any(self.entity_models_path.glob("entity_*.pt")):
                return True
        
        # Check for cluster models
        if self.cluster_models_path.exists():
            if any(self.cluster_models_path.glob("cluster_*.pt")):
                return True
                
        # Check for user models
        if self.user_models_path.exists():
            if any(self.user_models_path.glob("user_*.joblib")):
                return True
                
        # Check K-Means
        if self.kmeans_path.exists():
            return True
            
        return False
