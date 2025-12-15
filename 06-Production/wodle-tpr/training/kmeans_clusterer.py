import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path


class KMeansClusterer:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = RobustScaler()
        self.cluster_assignments = {}
        self.silhouette_score = None

    def fit(self, entity_metrics_dict):
        entity_ids = []
        feature_vectors = []

        for entity_id, metrics in entity_metrics_dict.items():
            entity_ids.append(entity_id)
            feature_vectors.append(metrics)

        X = np.array(feature_vectors)

        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=25)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        self.cluster_assignments = {
            entity_id: int(label)
            for entity_id, label in zip(entity_ids, cluster_labels)
        }

        # Calculate silhouette score for quality assessment
        if len(np.unique(cluster_labels)) > 1:
            self.silhouette_score = silhouette_score(X_scaled, cluster_labels)
        else:
            self.silhouette_score = None

        return self.cluster_assignments

    def predict(self, metrics_vector):
        if self.kmeans is None:
            return None

        X_scaled = self.scaler.transform(metrics_vector.reshape(1, -1))
        cluster = self.kmeans.predict(X_scaled)
        return int(cluster[0])

    def predict_from_windows(self, metrics_windows):
        """
        Predict cluster by aggregating multiple windows (consistent with training).

        This method should be used during detection to maintain consistency with
        how the model was trained (using mean aggregation across windows).

        Args:
            metrics_windows: np.array of shape (n_windows, n_features)
                            Multiple observation windows for an entity

        Returns:
            cluster_id: int - The predicted cluster ID
        """
        if self.kmeans is None:
            return None

        # Aggregate by mean (same as training)
        aggregated = np.mean(metrics_windows, axis=0).reshape(1, -1)
        X_scaled = self.scaler.transform(aggregated)
        cluster = self.kmeans.predict(X_scaled)
        return int(cluster[0])

    def predict_with_confidence(self, metrics_vector):
        """
        Predict cluster and return confidence metrics.

        Args:
            metrics_vector: np.array of shape (n_features,) or (1, n_features)

        Returns:
            dict with:
                - cluster: int - Predicted cluster ID
                - distances: list - Distances to all cluster centroids
                - confidence: float - Margin between closest and second-closest cluster
                - closest_distance: float - Distance to assigned cluster
                - second_closest_distance: float - Distance to next closest cluster
        """
        if self.kmeans is None:
            return None

        if metrics_vector.ndim == 1:
            metrics_vector = metrics_vector.reshape(1, -1)

        X_scaled = self.scaler.transform(metrics_vector)
        cluster = self.kmeans.predict(X_scaled)[0]
        distances = self.kmeans.transform(X_scaled)[0]

        sorted_distances = sorted(distances)
        confidence = sorted_distances[1] - sorted_distances[0] if len(sorted_distances) > 1 else 0.0

        return {
            'cluster': int(cluster),
            'distances': distances.tolist(),
            'confidence': float(confidence),
            'closest_distance': float(sorted_distances[0]),
            'second_closest_distance': float(sorted_distances[1]) if len(sorted_distances) > 1 else None
        }

    def get_cluster_sizes(self):
        if not self.cluster_assignments:
            return {}

        cluster_counts = {}
        for cluster_id in self.cluster_assignments.values():
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        return cluster_counts

    def save(self, path):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'cluster_assignments': self.cluster_assignments,
            'n_clusters': self.n_clusters,
            'silhouette_score': self.silhouette_score
        }, save_path)

    def load(self, path):
        data = joblib.load(path)
        self.kmeans = data['kmeans']
        self.scaler = data['scaler']
        self.cluster_assignments = data['cluster_assignments']
        self.n_clusters = data['n_clusters']
        self.silhouette_score = data.get('silhouette_score', None)
