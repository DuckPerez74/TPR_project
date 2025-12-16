"""
Training Orchestrator Module

This module orchestrates the training of all TPR models (L1 and L2).
Includes unified and per-window training pipelines.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import timedelta

from constants import L1_FEATURE_ORDER, L2_USER_FEATURES, L2_ROUTE_FEATURES
from core import OpenSearchClient
from data.metrics_fetcher import (
    get_unique_entities,
    fetch_entity_l1_metrics,
    fetch_entity_l1_metrics_all_windows,
    fetch_entity_l2_metrics,
    fetch_entity_l2_metrics_all
)
from training import KMeansClusterer, ModelTrainer, IsolationForestTrainer
from detection import ModelAssignmentCache


def train_l1_models_for_window(config, client, metrics_index, window_minutes, warmup_start, warmup_end):
    """
    Train L1 models (Auto-encoders + K-Means) for a specific observation window.

    Args:
        config: Configuration dictionary
        client: OpenSearch client
        metrics_index: Metrics index pattern
        window_minutes: Observation window size in minutes
        warmup_start: Start datetime of warmup period
        warmup_end: End datetime of warmup period

    Returns:
        Dict with training statistics or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Training L1 Models - {window_minutes}min Window")
    print(f"{'='*60}")

    models_base = Path(__file__).parent.parent / 'models'

    kmeans_start = warmup_start
    kmeans_end = warmup_start + timedelta(days=7)

    print(f"\n  Getting entities for K-Means clustering (first 7 days: {kmeans_start.date()} to {kmeans_end.date()})...")
    entities = get_unique_entities(client, metrics_index, kmeans_start, kmeans_end)

    if not entities:
        print(f"  WARNING: No entities found for {window_minutes}min window")
        return None

    print(f"\n  Step 1: K-Means Clustering ({window_minutes}min) - Processing {len(entities)} entities")
    print(f"  {'-'*56}")

    entity_mean_metrics = {}
    entities_with_data = []

    for i, entity_id in enumerate(entities):
        if (i + 1) % 100 == 0:
            print(f"    Processing entity {i+1}/{len(entities)} for K-Means...")

        samples = fetch_entity_l1_metrics(client, metrics_index, entity_id, kmeans_start, kmeans_end, window_minutes)

        if len(samples) >= 20:
            entity_mean_metrics[entity_id] = np.mean(samples, axis=0)
            entities_with_data.append(entity_id)

    print(f"    Entities with sufficient data for K-Means: {len(entities_with_data)}")

    if not entity_mean_metrics:
        print(f"  WARNING: No entities with sufficient data for K-Means")
        return None

    n_clusters = config.get('models', {}).get('n_clusters', 3)
    clusterer = KMeansClusterer(n_clusters=n_clusters, random_state=42)
    cluster_assignments = clusterer.fit(entity_mean_metrics)

    cluster_sizes = clusterer.get_cluster_sizes()
    print(f"    Cluster distribution: {cluster_sizes}")

    if clusterer.silhouette_score is not None:
        print(f"    Silhouette score: {clusterer.silhouette_score:.4f}")

    kmeans_path = models_base / f'kmeans_{window_minutes}min.pkl'
    clusterer.save(kmeans_path)
    print(f"    Saved K-means to: {kmeans_path}")

    print(f"\n  Saving cluster assignments to cache...")
    cache_db_path = Path(__file__).parent.parent / 'model_assignments.db'
    cache = ModelAssignmentCache(db_path=str(cache_db_path))

    for entity_id, cluster_id in cluster_assignments.items():
        cache.set_cluster_model(entity_id, cluster_id, confidence=None)

    print(f"    Saved {len(cluster_assignments)} cluster assignments")
    cluster_dist = cache.get_cluster_distribution()
    print(f"    Cluster distribution in cache: {cluster_dist}")

    print(f"\n  Step 2: Train Entity Auto-encoders ({window_minutes}min) - Full warmup period")
    print(f"  {'-'*56}")

    trainer = ModelTrainer(
        input_dim=len(L1_FEATURE_ORDER),
        encoding_dim=12,
        hidden_dim=30,
        batch_size=256,
        learning_rate=0.001,
        epochs=100
    )

    entity_models_path = models_base / 'entity_models' / f'{window_minutes}min'
    entity_models_trained = 0
    entity_models_list = []

    n_clusters = config.get('models', {}).get('n_clusters', 3)
    cluster_data = {cid: [] for cid in range(n_clusters)}

    for i, entity_id in enumerate(entities_with_data):
        if (i + 1) % 50 == 0:
            print(f"    Processing entity {i+1}/{len(entities_with_data)} for entity models...")

        samples = fetch_entity_l1_metrics(client, metrics_index, entity_id, warmup_start, warmup_end, window_minutes)

        if len(samples) >= 100:
            model_data = trainer.train_entity_model(entity_id, samples)

            if model_data:
                trainer.save_model(model_data, entity_id, entity_models_path)
                entity_models_trained += 1
                entity_models_list.append(entity_id)

        if entity_id in cluster_assignments and len(samples) > 0:
            cluster_id = cluster_assignments[entity_id]
            cluster_data[cluster_id].extend(samples.tolist())

    print(f"    Trained {entity_models_trained} entity models")

    print(f"\n  Updating cache with entity model assignments...")
    for entity_id in entity_models_list:
        cache.set_entity_model(entity_id, f"entity_{entity_id}")

    print(f"    Updated {len(entity_models_list)} entity assignments")

    stats = cache.get_stats()
    print(f"    Cache stats: {stats['entity_models']} entity models, {stats['cluster_models']} cluster assignments")

    print(f"\n  Step 3: Train Cluster Auto-encoders ({window_minutes}min)")
    print(f"  {'-'*56}")

    cluster_models_path = models_base / 'cluster_models' / f'{window_minutes}min'
    cluster_models_trained = 0
    MIN_CLUSTER_SAMPLES = 100

    for cluster_id, samples_list in cluster_data.items():
        samples = np.array(samples_list)
        if len(samples) < MIN_CLUSTER_SAMPLES:
            print(f"    WARNING: Cluster {cluster_id} has only {len(samples)} samples (min: {MIN_CLUSTER_SAMPLES}), skipping")
            continue

        model_data = trainer.train_cluster_model(cluster_id, samples)

        if model_data:
            trainer.save_cluster_model(model_data, cluster_id, cluster_models_path)
            cluster_models_trained += 1

    print(f"    Trained {cluster_models_trained} cluster models")

    return {
        'window': window_minutes,
        'entity_models': entity_models_trained,
        'cluster_models': cluster_models_trained,
        'kmeans_clusters': len(cluster_sizes)
    }


def train_l2_models_for_dimension(config, client, metrics_index, dimension, window_minutes, warmup_start, warmup_end):
    """
    Train L2 models (Isolation Forest) for a specific dimension.

    Args:
        config: Configuration dictionary
        client: OpenSearch client
        metrics_index: Metrics index pattern
        dimension: 'user' or 'route'
        window_minutes: Observation window size in minutes
        warmup_start: Start datetime of warmup period
        warmup_end: End datetime of warmup period

    Returns:
        Dict with training statistics or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Training L2 Models - {dimension.capitalize()} Dimension ({window_minutes}min)")
    print(f"{'='*60}")

    print(f"\n  Getting entities for L2 {dimension} training...")
    entities = get_unique_entities(client, metrics_index, warmup_start, warmup_end)

    if not entities:
        print(f"  WARNING: No entities found for L2 {dimension}")
        return None

    print(f"    Processing {len(entities)} entities for L2 {dimension} models...")

    dimension_metrics = {}

    for i, entity_id in enumerate(entities):
        if (i + 1) % 100 == 0:
            print(f"    Processing entity {i+1}/{len(entities)} for L2 {dimension}...")

        entity_dim_metrics = fetch_entity_l2_metrics(
            client, metrics_index, entity_id, warmup_start, warmup_end, dimension, window_minutes
        )

        for dim_value, samples in entity_dim_metrics.items():
            if dim_value not in dimension_metrics:
                dimension_metrics[dim_value] = []
            dimension_metrics[dim_value].extend(samples.tolist())

    print(f"\n  Filtering {dimension}s by minimum sample count...")
    filtered_metrics = {}
    for dim_value, samples_list in dimension_metrics.items():
        samples = np.array(samples_list)
        if len(samples) >= 20:
            filtered_metrics[dim_value] = samples

    print(f"    {dimension.capitalize()}s with >=20 samples: {len(filtered_metrics)}")

    if not filtered_metrics:
        print(f"  WARNING: No {dimension}s with sufficient data")
        return None

    print(f"\n  Training Isolation Forest models for {dimension}s...")
    print(f"  {'-'*56}")

    trainer = IsolationForestTrainer(n_estimators=100, contamination='auto', random_state=42)

    models_base = Path(__file__).parent.parent / 'models'
    dimension_models_path = models_base / f'{dimension}_models' / f'{window_minutes}min'
    models_trained = 0

    for dim_value, samples in filtered_metrics.items():
        model_data = trainer.train_user_model(dim_value, samples)

        if model_data:
            trainer.save_model(model_data, dim_value, dimension_models_path)
            models_trained += 1

    print(f"    Trained {models_trained} {dimension} models")

    return {
        'dimension': dimension,
        'window': window_minutes,
        'models_trained': models_trained
    }


def train_all_models_unified(config, train_l1=True, train_l2_user=True, train_l2_route=True, train_cluster=True):
    """
    Unified training pipeline that processes each entity once for all layers and windows.
    More efficient than the per-window approach.

    Args:
        config: Configuration dictionary
        train_l1: Whether to train L1 entity models
        train_l2_user: Whether to train L2 user models
        train_l2_route: Whether to train L2 route models
        train_cluster: Whether to train cluster models
    """
    print("\n" + "=" * 60)
    print("TPR Unified Model Training Pipeline")
    print("=" * 60)

    all_l2_dimensions = config.get('metrics', {}).get('layers', {}).get('L2', {}).get('dimensions', ['user', 'route'])
    l2_dimensions = []
    if train_l2_user and 'user' in all_l2_dimensions:
        l2_dimensions.append('user')
    if train_l2_route and 'route' in all_l2_dimensions:
        l2_dimensions.append('route')

    print("\nTraining Configuration:")
    print(f"  L1 Entity Models: {'✓ ENABLED' if train_l1 else '✗ SKIPPED'}")
    print(f"  L2 User Models: {'✓ ENABLED' if train_l2_user else '✗ SKIPPED'}")
    print(f"  L2 Route Models: {'✓ ENABLED' if train_l2_route else '✗ SKIPPED'}")
    print(f"  Cluster Models: {'✓ ENABLED' if train_cluster else '✗ SKIPPED'}")

    print("\nPhases that will execute:")
    phases_to_run = []
    if train_cluster or train_l1:
        phases_to_run.append("Phase 1: K-Means Clustering")
    if train_l1 or l2_dimensions:
        phases_to_run.append(f"Phase 2: Entity Processing ({', '.join([x for x in ['L1' if train_l1 else None] + [f'L2-{d}' for d in l2_dimensions] if x])})")
    if train_cluster:
        phases_to_run.append("Phase 3: Cluster Models")
    if l2_dimensions:
        phases_to_run.append(f"Phase 4: L2 Models ({', '.join(l2_dimensions)})")

    for phase in phases_to_run:
        print(f"  • {phase}")

    warmup_config = config.get('warmup', {})
    start_date_str = warmup_config.get('start_date', 'N/A')
    end_date_str = warmup_config.get('end_date', 'N/A')

    print(f"\nWarmup Period:")
    print(f"  Start: {start_date_str}")
    print(f"  End: {end_date_str}")

    from datetime import datetime
    try:
        warmup_start = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        warmup_end = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
    except Exception as e:
        print(f"ERROR: Failed to parse warmup dates: {str(e)}")
        sys.exit(1)

    client = OpenSearchClient.get_instance(config)
    base_index_name = config.get('indices', {}).get('metrics', {}).get('name', 'metrics-tpr')
    metrics_index = f"{base_index_name}*"

    print(f"\nMetrics Index Pattern: {metrics_index}")

    observation_windows = config.get('observation_windows', {}).get('enabled', [60, 30, 10])

    print(f"\nObservation Windows: {observation_windows}")
    print(f"L2 Dimensions to train: {l2_dimensions if l2_dimensions else 'None'}")

    models_base = Path(__file__).parent.parent / 'models'

    print(f"\n{'='*60}")
    print("Clearing Model Assignment Cache (Pre-Training)")
    print(f"{'='*60}")

    try:
        cache_db_path = Path(__file__).parent.parent / 'model_assignments.db'
        cache = ModelAssignmentCache(db_path=str(cache_db_path))
        cache.clear_all()
        print("  ✓ Model assignment cache cleared successfully")
    except Exception as e:
        print(f"  WARNING: Failed to clear cache: {str(e)}", file=sys.stderr)

    cluster_assignments = {}

    if train_cluster or train_l1:
        print(f"\n{'#'*60}")
        print("PHASE 1: K-Means Clustering (60min window, First 7 Days)")
        print(f"{'#'*60}")

        kmeans_start = warmup_start
        kmeans_end = warmup_start + timedelta(days=7)
        kmeans_window = 60

        print(f"\n{'='*60}")
        print(f"K-Means for {kmeans_window}min window")
        print(f"{'='*60}")

        entities = get_unique_entities(client, metrics_index, warmup_start, warmup_end)

        if not entities:
            print(f"  ERROR: No entities found")
            sys.exit(1)

        print(f"  Found {len(entities)} total entities")
        print(f"  Using first 7 days ({kmeans_start.date()} to {kmeans_end.date()}) for clustering")

        entity_mean_metrics = {}
        for i, entity_id in enumerate(entities):
            if (i + 1) % 100 == 0:
                print(f"    Processing entity {i+1}/{len(entities)}...")

            samples = fetch_entity_l1_metrics(client, metrics_index, entity_id, kmeans_start, kmeans_end, kmeans_window)

            if len(samples) >= 20:
                entity_mean_metrics[entity_id] = np.mean(samples, axis=0)

        if not entity_mean_metrics:
            print(f"  ERROR: No entities with sufficient data for K-Means")
            sys.exit(1)

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
    else:
        print(f"\n{'#'*60}")
        print("PHASE 1: K-Means Clustering - SKIPPED")
        print(f"{'#'*60}")

    if not train_l1 and not l2_dimensions:
        print(f"\n{'#'*60}")
        print("PHASE 2: Entity-by-Entity Training - SKIPPED (no L1 or L2 selected)")
        print(f"{'#'*60}")
        all_entities = []
    else:
        print(f"\n{'#'*60}")
        print("PHASE 2: Entity-by-Entity Training (L1 + L2, All Windows)")
        print(f"{'#'*60}")

        all_entities = get_unique_entities(client, metrics_index, warmup_start, warmup_end)

        if not all_entities:
            print("ERROR: No entities found!")
            sys.exit(1)

        print(f"\n  Total entities to process: {len(all_entities)}")

    l1_trainer = None
    l2_trainer = None

    if train_l1 or train_cluster:
        l1_trainer = ModelTrainer(
            input_dim=len(L1_FEATURE_ORDER),
            encoding_dim=12,
            hidden_dim=30,
            batch_size=256,
            learning_rate=0.001,
            epochs=100
        )

    if l2_dimensions:
        l2_trainer = IsolationForestTrainer(n_estimators=100, contamination='auto', random_state=42)

    cluster_data = {}
    if train_cluster or train_l1:
        n_clusters = config.get('models', {}).get('n_clusters', 3)
        cluster_data = {window: {cid: [] for cid in range(n_clusters)} for window in observation_windows}

    l2_accumulated_data = {}
    for dim in l2_dimensions:
        l2_accumulated_data[dim] = {}
        for window in observation_windows:
            l2_accumulated_data[dim][window] = {}

    l1_entity_models_trained = {window: 0 for window in observation_windows} if train_l1 else {}
    l2_models_trained = {dim: {window: 0 for window in observation_windows} for dim in l2_dimensions}

    for entity_idx, entity_id in enumerate(all_entities):
        if (entity_idx + 1) % 50 == 0:
            status_parts = []
            if train_l1:
                status_parts.append("L1")
            if l2_dimensions:
                status_parts.append(f"L2[{','.join(l2_dimensions)}]")
            print(f"\n  Processing entity {entity_idx+1}/{len(all_entities)} ({'/'.join(status_parts)}): {entity_id}")

        if train_l1 or train_cluster:
            l1_metrics_by_window = fetch_entity_l1_metrics_all_windows(
                client, metrics_index, entity_id, warmup_start, warmup_end, observation_windows
            )
        else:
            l1_metrics_by_window = {}

        if l2_dimensions:
            l2_metrics_all = fetch_entity_l2_metrics_all(
                client, metrics_index, entity_id, warmup_start, warmup_end, l2_dimensions, observation_windows
            )
        else:
            l2_metrics_all = {}

        if train_l1:
            for window in observation_windows:
                samples = l1_metrics_by_window.get(window, np.array([]))

                if len(samples) >= 100:
                    model_data = l1_trainer.train_entity_model(entity_id, samples)

                    if model_data:
                        entity_models_path = models_base / 'entity_models' / f'{window}min'
                        l1_trainer.save_model(model_data, entity_id, entity_models_path)
                        l1_entity_models_trained[window] += 1

                        cache.set_entity_model(entity_id, f"entity_{entity_id}")

                if train_cluster and len(samples) > 0 and entity_id in cluster_assignments:
                    cluster_id = cluster_assignments[entity_id]
                    cluster_data[window][cluster_id].extend(samples.tolist())

        for dim in l2_dimensions:
            for window in observation_windows:
                dim_metrics = l2_metrics_all.get(dim, {}).get(window, {})

                for dim_value, samples in dim_metrics.items():
                    if dim_value not in l2_accumulated_data[dim][window]:
                        l2_accumulated_data[dim][window][dim_value] = []
                    l2_accumulated_data[dim][window][dim_value].extend(samples.tolist())

    if all_entities:
        print(f"\n  Entity processing complete!")
        if train_l1:
            print(f"  L1 Entity models trained: {l1_entity_models_trained}")
        if l2_dimensions:
            print(f"  L2 data accumulated for dimensions: {l2_dimensions}")

    cluster_models_trained = {window: 0 for window in observation_windows}

    if train_cluster:
        print(f"\n{'#'*60}")
        print("PHASE 3: Cluster Model Training (L1)")
        print(f"{'#'*60}")

        MIN_CLUSTER_SAMPLES = 100

        for window in observation_windows:
            print(f"\n  Training cluster models for {window}min window...")

            cluster_models_path = models_base / 'cluster_models' / f'{window}min'

            for cluster_id, samples_list in cluster_data[window].items():
                samples = np.array(samples_list)

                if len(samples) < MIN_CLUSTER_SAMPLES:
                    print(f"    WARNING: Cluster {cluster_id} has only {len(samples)} samples (min: {MIN_CLUSTER_SAMPLES}), skipping")
                    continue

                model_data = l1_trainer.train_cluster_model(cluster_id, samples)

                if model_data:
                    l1_trainer.save_cluster_model(model_data, cluster_id, cluster_models_path)
                    cluster_models_trained[window] += 1

            print(f"    Trained {cluster_models_trained[window]} cluster models for {window}min")
    else:
        print(f"\n{'#'*60}")
        print("PHASE 3: Cluster Model Training - SKIPPED")
        print(f"{'#'*60}")

    if l2_dimensions:
        print(f"\n{'#'*60}")
        print("PHASE 4: L2 Dimension Model Training")
        print(f"{'#'*60}")

        for dim in l2_dimensions:
            for window in observation_windows:
                print(f"\n  Training L2 {dim} models for {window}min window...")

                dimension_models_path = models_base / f'{dim}_models' / f'{window}min'

                for dim_value, samples_list in l2_accumulated_data[dim][window].items():
                    samples = np.array(samples_list)

                    if len(samples) < 20:
                        continue

                    model_data = l2_trainer.train_user_model(dim_value, samples)

                    if model_data:
                        l2_trainer.save_model(model_data, dim_value, dimension_models_path)
                        l2_models_trained[dim][window] += 1

                print(f"    Trained {l2_models_trained[dim][window]} {dim} models for {window}min")
    else:
        print(f"\n{'#'*60}")
        print("PHASE 4: L2 Dimension Model Training - SKIPPED")
        print(f"{'#'*60}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if train_cluster or train_l1:
        print("\nL1 Models (Auto-encoders + K-Means):")
        if train_cluster:
            print(f"  K-Means: 1 model (60min window, 3 clusters)")
        if train_l1:
            for window in observation_windows:
                print(f"  {window}min window:")
                print(f"    - Entity models: {l1_entity_models_trained[window]}")
                if train_cluster:
                    print(f"    - Cluster models: {cluster_models_trained[window]}")

    if l2_dimensions:
        print("\nL2 Models (Isolation Forest):")
        for dim in l2_dimensions:
            for window in observation_windows:
                print(f"  {dim.capitalize()} dimension - {window}min window:")
                print(f"    - Models trained: {l2_models_trained[dim][window]}")

    print("\nModels Location:")
    print(f"  Base: {models_base}")
    print(f"  L1 K-Means: models/kmeans_60min.pkl (shared by all windows)")
    print(f"  L1 Entity: models/entity_models/{{60|30|10}}min/")
    print(f"  L1 Cluster: models/cluster_models/{{60|30|10}}min/")
    print(f"  L2 User: models/user_models/{{60|30|10}}min/")
    print(f"  L2 Route: models/route_models/{{60|30|10}}min/")
    print(f"\nCache: model_assignments.db (in root)")


def train_models(config):
    """
    Main training pipeline for all L1 and L2 models (per-window approach).

    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("TPR Model Training Pipeline")
    print("=" * 60)

    warmup_config = config.get('warmup', {})
    start_date_str = warmup_config.get('start_date', 'N/A')
    end_date_str = warmup_config.get('end_date', 'N/A')

    print(f"\nWarmup Period:")
    print(f"  Start: {start_date_str}")
    print(f"  End: {end_date_str}")

    from datetime import datetime
    try:
        warmup_start = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        warmup_end = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
    except Exception as e:
        print(f"ERROR: Failed to parse warmup dates: {str(e)}")
        sys.exit(1)

    client = OpenSearchClient.get_instance(config)
    base_index_name = config.get('indices', {}).get('metrics', {}).get('name', 'metrics-tpr')
    metrics_index = f"{base_index_name}*"

    print(f"\nMetrics Index Pattern: {metrics_index}")

    try:
        cat_response = client.cat.indices(index=metrics_index, format='json', ignore_unavailable=True)
        if cat_response:
            print(f"Matching Indices Found:")
            for idx in cat_response:
                print(f"  - {idx['index']} ({idx.get('docs.count', 'N/A')} docs, {idx.get('store.size', 'N/A')})")
        else:
            print(f"WARNING: No indices found matching pattern '{metrics_index}'")
    except Exception as e:
        print(f"WARNING: Could not list indices: {str(e)}")

    observation_windows = config.get('observation_windows', {}).get('enabled', [60, 30, 10])
    print(f"\nObservation Windows: {observation_windows}")

    l2_dimensions = config.get('metrics', {}).get('layers', {}).get('L2', {}).get('dimensions', ['user', 'route'])
    print(f"L2 Dimensions: {l2_dimensions}")

    print(f"\n{'='*60}")
    print("Clearing Model Assignment Cache (Pre-Training)")
    print(f"{'='*60}")

    try:
        cache_db_path = Path(__file__).parent.parent / 'model_assignments.db'
        cache = ModelAssignmentCache(db_path=str(cache_db_path))
        cache.clear_all()
        print("  ✓ Model assignment cache cleared successfully")
    except Exception as e:
        print(f"  WARNING: Failed to clear cache: {str(e)}", file=sys.stderr)

    print(f"\n{'#'*60}")
    print("LAYER 1 (L1) - Entity & Cluster Models")
    print(f"{'#'*60}")

    l1_results = []
    for window in observation_windows:
        result = train_l1_models_for_window(config, client, metrics_index, window, warmup_start, warmup_end)
        if result:
            l1_results.append(result)

    print(f"\n{'#'*60}")
    print("LAYER 2 (L2) - Dimension Models (Isolation Forest)")
    print(f"{'#'*60}")

    l2_results = []
    for dimension in l2_dimensions:
        for window in observation_windows:
            result = train_l2_models_for_dimension(config, client, metrics_index, dimension, window, warmup_start, warmup_end)
            if result:
                l2_results.append(result)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    print("\nL1 Models (Auto-encoders + K-Means):")
    for result in l1_results:
        print(f"  {result['window']}min window:")
        print(f"    - Entity models: {result['entity_models']}")
        print(f"    - Cluster models: {result['cluster_models']}")
        print(f"    - K-Means clusters: {result['kmeans_clusters']}")

    print("\nL2 Models (Isolation Forest):")
    for result in l2_results:
        print(f"  {result['dimension'].capitalize()} dimension - {result['window']}min window:")
        print(f"    - Models trained: {result['models_trained']}")

    print("\nModels Location:")
    models_base = Path(__file__).parent.parent / 'models'
    print(f"  Base: {models_base}")
    print(f"  L1 Entity: models/entity_models/{{60|30|10}}min/")
    print(f"  L1 Cluster: models/cluster_models/{{60|30|10}}min/")
    print(f"  L1 K-Means: models/kmeans_{{60|30|10}}min.pkl")
    print(f"  L2 User: models/user_models/{{60|30|10}}min/")
    print(f"  L2 Route: models/route_models/{{60|30|10}}min/")
