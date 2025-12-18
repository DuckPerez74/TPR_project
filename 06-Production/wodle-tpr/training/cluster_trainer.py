import numpy as np
from pathlib import Path
from datetime import timedelta

from constants import L1_FEATURE_ORDER
from data.metrics_fetcher import (
    get_unique_entities,
    fetch_all_l1_metrics_by_window,
    fetch_cluster_l1_metrics
)
from training import KMeansClusterer, ModelTrainer
from detection import ModelAssignmentCache


def train_kmeans_clustering(config, client, metrics_index, warmup_start, warmup_end, models_base):
    print(f"\n{'#'*60}")
    print("PHASE 1: K-Means Clustering (60min window, First 7 Days)")
    print(f"{'#'*60}")

    kmeans_start = warmup_start
    kmeans_end = warmup_start + timedelta(days=7)
    kmeans_window = 60

    print(f"\n{'='*60}")
    print(f"K-Means for {kmeans_window}min window")
    print(f"{'='*60}")

    print(f"  [PHASE 1/3] Fetching entities from OpenSearch...")
    entities = get_unique_entities(client, metrics_index, warmup_start, warmup_end)

    if not entities:
        print(f"  ERROR: No entities found")
        raise ValueError("No entities found for K-Means clustering")

    print(f"  ✓ Found {len(entities)} total entities")
    print(f"  Using first 7 days ({kmeans_start.date()} to {kmeans_end.date()}) for clustering")

    print(f"\n  [PHASE 1/3] Fetching L1 metrics for K-Means clustering...")
    print(f"    → Using BATCH QUERY (1 query for all {len(entities)} entities)...")

    all_entity_metrics = fetch_all_l1_metrics_by_window(
        client, metrics_index, kmeans_start, kmeans_end, kmeans_window
    )

    print(f"    → Fetched data for {len(all_entity_metrics)} entities, computing means...")

    entity_mean_metrics = {}
    for entity_id in entities:
        samples = all_entity_metrics.get(entity_id, np.array([]))
        if len(samples) >= 20:
            entity_mean_metrics[entity_id] = np.mean(samples, axis=0)

    if not entity_mean_metrics:
        print(f"  ERROR: No entities with sufficient data for K-Means")
        raise ValueError("No entities with sufficient data for K-Means")

    print(f"    Entities with ≥20 samples: {len(entity_mean_metrics)}")

    n_clusters = config.get('models', {}).get('n_clusters', 3)
    clusterer = KMeansClusterer(n_clusters=n_clusters, random_state=42)
    cluster_assignments = clusterer.fit(entity_mean_metrics)

    cluster_sizes = clusterer.get_cluster_sizes()
    print(f"    Cluster distribution: {cluster_sizes}")

    if clusterer.silhouette_score is not None:
        print(f"    Silhouette score: {clusterer.silhouette_score:.4f}")

    kmeans_path = models_base / f'kmeans_{kmeans_window}min.pkl'
    clusterer.save(kmeans_path)
    print(f"    Saved K-means to: {kmeans_path}")

    cache_db_path = Path(__file__).parent.parent / 'model_assignments.db'
    cache = ModelAssignmentCache(db_path=str(cache_db_path))

    for entity_id, cluster_id in cluster_assignments.items():
        cache.set_cluster_model(entity_id, cluster_id, confidence=None)

    print(f"    Saved {len(cluster_assignments)} cluster assignments to cache")
    print(f"    These assignments will be used for ALL observation windows")

    return cluster_assignments


def train_cluster_models(config, client, metrics_index, warmup_start, warmup_end,
                        cluster_assignments, observation_windows, models_base):
    print(f"\n{'#'*60}")
    print("PHASE 3: Cluster Model Training (L1)")
    print(f"{'#'*60}")
    print("  Using DIRECT QUERIES to OpenSearch (no memory accumulation)")

    MIN_CLUSTER_SAMPLES = 100

    l1_trainer = ModelTrainer(
        input_dim=len(L1_FEATURE_ORDER),
        encoding_dim=12,
        hidden_dim=30,
        batch_size=256,
        learning_rate=0.001,
        epochs=100
    )

    n_clusters = config.get('models', {}).get('n_clusters', 3)

    cluster_entities = {cid: [] for cid in range(n_clusters)}
    for entity_id, cluster_id in cluster_assignments.items():
        cluster_entities[cluster_id].append(entity_id)

    cluster_models_trained = {}

    for window in observation_windows:
        print(f"\n  [PHASE 3/3] Training cluster models for {window}min window...")

        cluster_models_path = models_base / 'cluster_models' / f'{window}min'
        cluster_models_trained[window] = 0

        for cluster_id in range(n_clusters):
            entities_in_cluster = cluster_entities[cluster_id]

            if not entities_in_cluster:
                print(f"    ⚠ Cluster {cluster_id}: no entities assigned, skipping")
                continue

            print(f"    ⚙ Cluster {cluster_id}: Fetching L1 metrics for {len(entities_in_cluster)} entities...")

            samples = fetch_cluster_l1_metrics(
                client, metrics_index, entities_in_cluster,
                warmup_start, warmup_end, window
            )

            if len(samples) < MIN_CLUSTER_SAMPLES:
                print(f"    ⚠ Cluster {cluster_id}: only {len(samples)} samples (min: {MIN_CLUSTER_SAMPLES}), skipping")
                continue

            print(f"    ⚙ Training cluster {cluster_id} model ({len(samples)} samples)...")
            model_data = l1_trainer.train_cluster_model(cluster_id, samples)

            if model_data:
                l1_trainer.save_cluster_model(model_data, cluster_id, cluster_models_path)
                cluster_models_trained[window] += 1
                print(f"    ✓ Cluster {cluster_id} model saved")

        print(f"    ✓ Trained {cluster_models_trained[window]} cluster models for {window}min")

    return cluster_models_trained
