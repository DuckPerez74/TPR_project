import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class KMeansClusterer:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_assignments = {}

    def fit(self, entity_metrics_dict):
        entity_ids = []
        feature_vectors = []

        for entity_id, metrics in entity_metrics_dict.items():
            entity_ids.append(entity_id)
            feature_vectors.append(metrics)

        X = np.array(feature_vectors)

        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        self.cluster_assignments = {
            entity_id: int(label)
            for entity_id, label in zip(entity_ids, cluster_labels)
        }

        return self.cluster_assignments

    def predict(self, metrics_vector):
        if self.kmeans is None:
            return None

        X_scaled = self.scaler.transform(metrics_vector.reshape(1, -1))
        cluster = self.kmeans.predict(X_scaled)
        return int(cluster[0])

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

        with open(save_path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'cluster_assignments': self.cluster_assignments,
                'n_clusters': self.n_clusters
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.scaler = data['scaler']
            self.cluster_assignments = data['cluster_assignments']
            self.n_clusters = data['n_clusters']
