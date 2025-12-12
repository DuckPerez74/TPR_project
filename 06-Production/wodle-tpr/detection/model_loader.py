import pickle
import json
import torch
from pathlib import Path
import joblib
from training.autoencoder import AutoEncoder


class ModelLoader:
    def __init__(self, config: dict):
        models_config = config.get('models', {})
        base_path = Path(__file__).parent.parent

        self.entity_models_path = base_path / models_config.get('entity_models_path', 'models/entity_models')
        self.cluster_models_path = base_path / models_config.get('cluster_models_path', 'models/cluster_models')
        self.user_models_path = base_path / models_config.get('user_models_path', 'models/user_models')
        self.kmeans_path = base_path / 'models' / 'kmeans_clusterer.pkl'

        self.threshold_fallback = config.get('detection', {}).get('threshold_fallback', 0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.entity_models = {}
        self.cluster_models = {}
        self.user_models = {}
        self.entity_scalers = {}
        self.cluster_scalers = {}
        self.user_scalers = {}
        self.entity_thresholds = {}
        self.cluster_thresholds = {}
        self.kmeans_clusterer = None

        self._load_all()

    def _load_all(self):
        self._load_entity_models()
        self._load_cluster_models()
        self._load_user_models()
        self._load_kmeans()

    def _load_entity_models(self):
        if not self.entity_models_path.exists():
            return

        for model_file in self.entity_models_path.glob("entity_*.pt"):
            try:
                entity_id = model_file.stem.replace("entity_", "")

                checkpoint = torch.load(model_file, map_location=self.device)
                model = AutoEncoder(
                    checkpoint['input_dim'],
                    checkpoint['encoding_dim'],
                    checkpoint['hidden_dim']
                ).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.entity_models[entity_id] = model

                scaler_file = model_file.parent / f"entity_{entity_id}_scaler.pkl"
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.entity_scalers[entity_id] = pickle.load(f)

                threshold_file = model_file.parent / f"entity_{entity_id}_thresholds.json"
                if threshold_file.exists():
                    with open(threshold_file, 'r') as f:
                        thresholds = json.load(f)
                        self.entity_thresholds[entity_id] = thresholds.get('p95', self.threshold_fallback)
            except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError) as e:
                import sys
                print(f"WARNING: Failed to load entity model {entity_id}: {str(e)}", file=sys.stderr)
                continue

    def _load_cluster_models(self):
        if not self.cluster_models_path.exists():
            return

        for model_file in self.cluster_models_path.glob("cluster_*.pt"):
            try:
                cluster_id = int(model_file.stem.split('_')[1])

                checkpoint = torch.load(model_file, map_location=self.device)
                model = AutoEncoder(
                    checkpoint['input_dim'],
                    checkpoint['encoding_dim'],
                    checkpoint['hidden_dim']
                ).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.cluster_models[cluster_id] = model

                scaler_file = model_file.parent / f"cluster_{cluster_id}_scaler.pkl"
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.cluster_scalers[cluster_id] = pickle.load(f)

                threshold_file = model_file.parent / f"cluster_{cluster_id}_thresholds.json"
                if threshold_file.exists():
                    with open(threshold_file, 'r') as f:
                        thresholds = json.load(f)
                        self.cluster_thresholds[cluster_id] = thresholds.get('p95', self.threshold_fallback)
            except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError, ValueError) as e:
                import sys
                print(f"WARNING: Failed to load cluster model {cluster_id}: {str(e)}", file=sys.stderr)
                continue

    def _load_user_models(self):
        if not self.user_models_path.exists():
            return

        for model_file in self.user_models_path.glob("user_*.joblib"):
            try:
                user_id_safe = model_file.stem[5:] 
                
                model = joblib.load(model_file)
                self.user_models[user_id_safe] = model

                scaler_file = model_file.parent / f"user_{user_id_safe}_scaler.joblib"
                if scaler_file.exists():
                    self.user_scalers[user_id_safe] = joblib.load(scaler_file)

            except (FileNotFoundError, KeyError, RuntimeError, ValueError) as e:
                import sys
                print(f"WARNING: Failed to load user model {model_file.stem}: {str(e)}", file=sys.stderr)
                continue

    def _load_kmeans(self):
        if not self.kmeans_path.exists():
            return

        try:
            from training.kmeans_clusterer import KMeansClusterer
            self.kmeans_clusterer = KMeansClusterer()
            self.kmeans_clusterer.load(self.kmeans_path)
        except (FileNotFoundError, ImportError, pickle.UnpicklingError) as e:
            import sys
            print(f"WARNING: Failed to load K-means clusterer: {str(e)}", file=sys.stderr)

    def get_entity_model(self, entity_id: str):
        return self.entity_models.get(entity_id)

    def get_entity_scaler(self, entity_id: str):
        return self.entity_scalers.get(entity_id)

    def get_cluster_model(self, cluster_id: int):
        return self.cluster_models.get(cluster_id)

    def get_cluster_scaler(self, cluster_id: int):
        return self.cluster_scalers.get(cluster_id)

    def get_entity_threshold(self, entity_id: str) -> float:
        return self.entity_thresholds.get(entity_id, self.threshold_fallback)

    def get_cluster_threshold(self, cluster_id: int) -> float:
        return self.cluster_thresholds.get(cluster_id, self.threshold_fallback)

    def has_entity_model(self, entity_id: str) -> bool:
        return entity_id in self.entity_models

    def get_user_model(self, user_id: str):
        # We need to sanitize the input user_id to match the file storage convention
        safe_user_id = "".join([c if c.isalnum() else "_" for c in user_id])
        return self.user_models.get(safe_user_id)

    def get_user_scaler(self, user_id: str):
        safe_user_id = "".join([c if c.isalnum() else "_" for c in user_id])
        return self.user_scalers.get(safe_user_id)

    def get_available_clusters(self) -> list:
        return list(self.cluster_models.keys())

    def predict_cluster(self, metrics_vector):
        if self.kmeans_clusterer is None:
            clusters = self.get_available_clusters()
            return clusters[0] if clusters else None

        return self.kmeans_clusterer.predict(metrics_vector)
