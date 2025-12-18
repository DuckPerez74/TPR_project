"""
Training Orchestrator Module

This module orchestrates the training of all TPR models (L1 and L2).
Includes unified and per-window training pipelines.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

from constants import L1_FEATURE_ORDER, L2_USER_FEATURES, L2_ROUTE_FEATURES
from core import OpenSearchClient
from data.metrics_fetcher import (
    get_unique_entities,
    fetch_entity_l1_metrics,
    fetch_entity_l2_metrics,
    fetch_all_l1_metrics_by_window,
    fetch_all_l2_metrics_by_dimension_window
)
from training import KMeansClusterer, ModelTrainer, IsolationForestTrainer
from detection import ModelAssignmentCache


# Helper functions for parallel training

def _get_optimal_workers(task_type='cpu'):
    """
    Determine optimal number of workers based on CPU count and instance type.

    CRITICAL: Limited to max 4 workers to prevent memory exhaustion when training
    thousands of models. Each worker duplicates data and models in RAM.

    Args:
        task_type: 'cpu' for CPU-bound (ProcessPoolExecutor), 'io' for I/O-bound (ThreadPoolExecutor)

    Returns:
        int: Recommended number of workers (max 4)
    """
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # HARD LIMIT: Max 4 workers to balance speed and memory usage
    # Each worker processes one entity at a time, which can be memory-intensive
    max_workers = 4

    print(f"    → Detected {cpu_count} CPUs")
    print(f"    → Using {max_workers} parallel workers (memory-optimized for training thousands of models)")

    return max_workers


def _train_single_entity_model(args):
    """Helper for parallel entity model training. Returns (entity_id, model_data, window)."""
    entity_id, samples, window, config = args

    if len(samples) < 100:
        return entity_id, None, window

    trainer = ModelTrainer(
        input_dim=len(L1_FEATURE_ORDER),
        encoding_dim=12,
        hidden_dim=30,
        batch_size=256,
        learning_rate=0.001,
        epochs=100
    )

    model_data = trainer.train_entity_model(entity_id, samples)
    return entity_id, model_data, window


def _train_single_l2_model(args):
    """Helper for parallel L2 model training. Returns (dim_value, model_data, dimension, window)."""
    dim_value, samples, dimension, window = args

    if len(samples) < 20:
        return dim_value, None, dimension, window

    trainer = IsolationForestTrainer(n_estimators=100, contamination='auto', random_state=42)
    model_data = trainer.train_user_model(dim_value, samples)
    return dim_value, model_data, dimension, window


def _process_single_entity_all_windows(args):
    """
    Process a SINGLE entity: fetch data, train models for ALL windows and layers.
    This is the core unit of work - one entity at a time to conserve memory.

    Args:
        args: Tuple of (entity_id, config, client, warmup_start, warmup_end,
                       observation_windows, l2_dimensions, models_base,
                       cluster_assignments, train_l1, train_cluster)

    Returns:
        Dict with results and cluster data samples
    """
    import gc
    import warnings
    import time
    from datetime import datetime

    # Suppress warnings in worker process
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
    warnings.filterwarnings('ignore', message='.*Unverified HTTPS request.*')
    warnings.filterwarnings('ignore', message='.*SSL.*', category=Warning)

    entity_id, config, metrics_index, warmup_start, warmup_end, observation_windows, \
        l2_dimensions, models_base, cluster_assignments, train_l1, train_cluster = args

    # Import inside worker to avoid pickling issues
    from core import OpenSearchClient
    client = OpenSearchClient.get_instance(config)

    def _timestamp():
        """Get current timestamp for logging"""
        return datetime.now().strftime("%H:%M:%S")

    def _duration(start_time):
        """Calculate duration in human-readable format"""
        elapsed = time.time() - start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            return f"{elapsed/60:.1f}min"
        else:
            return f"{elapsed/3600:.1f}h"

    entity_start = time.time()
    print(f"    [{_timestamp()}] [Entity {entity_id}] Starting processing (all windows: {observation_windows})")

    results = {
        'entity_id': entity_id,
        'l1_models_trained': {},
        'l2_models_trained': {dim: {} for dim in l2_dimensions},
        'cluster_samples': {}  # Will accumulate samples for cluster training
    }

    for window in observation_windows:
        results['l1_models_trained'][window] = 0
        for dim in l2_dimensions:
            results['l2_models_trained'][dim][window] = 0
        results['cluster_samples'][window] = []

        # L1 Fetch
        t0 = time.time()
        print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Fetching L1 metrics...")

        # ========== L1: Autoencoder ==========
        if train_l1 or train_cluster:
            samples = fetch_entity_l1_metrics(client, metrics_index, entity_id,
                                             warmup_start, warmup_end, window)

            print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Got {len(samples)} L1 samples (fetch: {_duration(t0)})")

            if train_l1 and len(samples) >= 100:
                t1 = time.time()
                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Training L1 autoencoder...")
                trainer = ModelTrainer(
                    input_dim=len(L1_FEATURE_ORDER),
                    encoding_dim=12,
                    hidden_dim=30,
                    batch_size=256,
                    learning_rate=0.001,
                    epochs=100
                )
                model_data = trainer.train_entity_model(entity_id, samples)

                if model_data:
                    entity_models_path = models_base / 'entity_models' / f'{window}min'
                    trainer.save_model(model_data, entity_id, entity_models_path)
                    results['l1_models_trained'][window] = 1
                    print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: ✓ L1 autoencoder trained in {_duration(t1)}")
            elif train_l1:
                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Skipping L1 (insufficient samples: {len(samples)}/100)")

            # Accumulate for cluster training (save cluster_id + mean of samples)
            if train_cluster and len(samples) > 0 and entity_id in cluster_assignments:
                cluster_id = cluster_assignments[entity_id]
                # Store only mean to save memory
                results['cluster_samples'][window].append({
                    'cluster_id': cluster_id,
                    'samples': samples.copy()  # Keep samples for cluster training
                })

        # ========== L2: Isolation Forest - User ==========
        if 'user' in l2_dimensions:
            t2 = time.time()
            print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Fetching L2 user metrics...")
            user_metrics = fetch_entity_l2_metrics(client, metrics_index, entity_id,
                                                  warmup_start, warmup_end, 'user', window)

            if user_metrics:
                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Training {len(user_metrics)} user models (fetch: {_duration(t2)})...")
                t3 = time.time()
                trainer = IsolationForestTrainer(n_estimators=100, contamination='auto', random_state=42)
                user_models_path = models_base / 'user_models' / f'{window}min'

                total_users = len(user_metrics)
                trained_count = 0
                for idx, (user_value, samples) in enumerate(user_metrics.items(), 1):
                    if len(samples) >= 20:
                        model_data = trainer.train_user_model(user_value, samples)
                        if model_data:
                            trainer.save_model(model_data, user_value, user_models_path)
                            results['l2_models_trained']['user'][window] += 1
                            trained_count += 1

                    # Show progress every 100 models or at milestones
                    if total_users > 200 and (idx % 100 == 0 or idx == total_users):
                        progress_pct = (idx / total_users) * 100
                        print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min:   -> User models: {idx}/{total_users} ({progress_pct:.0f}%, {trained_count} trained)")

                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: ✓ {results['l2_models_trained']['user'][window]} user models trained in {_duration(t3)}")

        # ========== L2: Isolation Forest - Route ==========
        if 'route' in l2_dimensions:
            t4 = time.time()
            print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Fetching L2 route metrics...")
            route_metrics = fetch_entity_l2_metrics(client, metrics_index, entity_id,
                                                   warmup_start, warmup_end, 'route', window)

            if route_metrics:
                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Training {len(route_metrics)} route models (fetch: {_duration(t4)})...")
                t5 = time.time()
                trainer = IsolationForestTrainer(n_estimators=100, contamination='auto', random_state=42)
                route_models_path = models_base / 'route_models' / f'{window}min'

                total_routes = len(route_metrics)
                trained_count = 0
                for idx, (route_value, samples) in enumerate(route_metrics.items(), 1):
                    if len(samples) >= 20:
                        model_data = trainer.train_user_model(route_value, samples)
                        if model_data:
                            trainer.save_model(model_data, route_value, route_models_path)
                            results['l2_models_trained']['route'][window] += 1
                            trained_count += 1

                    # Show progress every 100 models or at milestones
                    if total_routes > 200 and (idx % 100 == 0 or idx == total_routes):
                        progress_pct = (idx / total_routes) * 100
                        print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min:   -> Route models: {idx}/{total_routes} ({progress_pct:.0f}%, {trained_count} trained)")

                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: ✓ {results['l2_models_trained']['route'][window]} route models trained in {_duration(t5)}")

    # Explicit memory cleanup
    total_time = _duration(entity_start)
    total_l1 = sum(results['l1_models_trained'].values())
    total_user = sum(results['l2_models_trained'].get('user', {}).values())
    total_route = sum(results['l2_models_trained'].get('route', {}).values())

    print(f"    [{_timestamp()}] [Entity {entity_id}] ✓ Completed all windows in {total_time} (L1:{total_l1}, User:{total_user}, Route:{total_route}) - clearing memory")
    gc.collect()

    return results


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

        print(f"  [PHASE 1/3] Fetching entities from OpenSearch...")
        entities = get_unique_entities(client, metrics_index, warmup_start, warmup_end)

        if not entities:
            print(f"  ERROR: No entities found")
            sys.exit(1)

        print(f"  ✓ Found {len(entities)} total entities")
        print(f"  Using first 7 days ({kmeans_start.date()} to {kmeans_end.date()}) for clustering")

        print(f"\n  [PHASE 1/3] Fetching L1 metrics for K-Means clustering...")
        print(f"    → Using BATCH QUERY (1 query for all {len(entities)} entities)...")

        # OPTIMIZATION: Fetch ALL entities at once (1 query instead of N queries)
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

        print(f"  [PHASE 2/3] Fetching entities list from OpenSearch...")
        all_entities = get_unique_entities(client, metrics_index, warmup_start, warmup_end)

        if not all_entities:
            print("ERROR: No entities found!")
            sys.exit(1)

        print(f"  ✓ Total entities to process: {len(all_entities)}")
        print(f"  ⚙  Processing ONE entity at a time (all windows + all layers per entity)")
        print(f"  ⚙  Memory will be freed after each entity")
        print(f"  ⚙  Using max {_get_optimal_workers()} parallel workers")

    # Initialize counters and cluster data accumulator
    cluster_data = {}
    if train_cluster:
        n_clusters = config.get('models', {}).get('n_clusters', 3)
        cluster_data = {window: {cid: [] for cid in range(n_clusters)} for window in observation_windows}

    l1_entity_models_trained = {window: 0 for window in observation_windows} if train_l1 else {}
    l2_models_trained = {dim: {window: 0 for window in observation_windows} for dim in l2_dimensions}

    # NEW APPROACH: Process ONE entity at a time (all windows + all layers)
    if all_entities:
        print(f"\n  [PHASE 2/3] Processing {len(all_entities)} entities (one at a time)...")

        # Prepare arguments for parallel processing
        entity_jobs = []
        for entity_id in all_entities:
            job_args = (
                entity_id, config, metrics_index, warmup_start, warmup_end,
                observation_windows, l2_dimensions, models_base,
                cluster_assignments, train_l1, train_cluster
            )
            entity_jobs.append(job_args)

        max_workers = _get_optimal_workers(task_type='cpu')

        print(f"  ⚙  Using {max_workers} parallel workers to process entities")
        print(f"  ⚙  Each worker processes one entity completely before moving to next")
        print(f"")

        # Process entities with limited parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_entity_all_windows, job): job[0] for job in entity_jobs}

            completed = 0
            total_l1_trained = 0
            total_l2_user_trained = 0
            total_l2_route_trained = 0

            for future in as_completed(futures):
                entity_id = futures[future]
                result = future.result()

                # Count models trained for this entity
                entity_l1_count = sum(result['l1_models_trained'].values())
                entity_l2_user_count = sum(result['l2_models_trained'].get('user', {}).values())
                entity_l2_route_count = sum(result['l2_models_trained'].get('route', {}).values())

                total_l1_trained += entity_l1_count
                total_l2_user_trained += entity_l2_user_count
                total_l2_route_trained += entity_l2_route_count

                # Aggregate results
                for window in observation_windows:
                    l1_entity_models_trained[window] += result['l1_models_trained'].get(window, 0)
                    for dim in l2_dimensions:
                        l2_models_trained[dim][window] += result['l2_models_trained'][dim].get(window, 0)

                    # Accumulate cluster samples (if any)
                    if train_cluster and window in result['cluster_samples']:
                        for sample_data in result['cluster_samples'][window]:
                            cluster_id = sample_data['cluster_id']
                            samples = sample_data['samples']
                            cluster_data[window][cluster_id].extend(samples.tolist())

                completed += 1
                progress_pct = (completed / len(all_entities)) * 100

                # Show detailed progress
                models_summary = []
                if entity_l1_count > 0:
                    models_summary.append(f"L1:{entity_l1_count}")
                if entity_l2_user_count > 0:
                    models_summary.append(f"User:{entity_l2_user_count}")
                if entity_l2_route_count > 0:
                    models_summary.append(f"Route:{entity_l2_route_count}")

                models_str = ", ".join(models_summary) if models_summary else "no models"

                print(f"    → [{completed}/{len(all_entities)} = {progress_pct:.1f}%] Entity {entity_id}: {models_str} | Total so far: L1={total_l1_trained}, User={total_l2_user_trained}, Route={total_l2_route_trained}")

        print(f"\n  ✓ Entity processing complete!")
        if train_l1:
            print(f"  ✓ L1 Entity models trained: {l1_entity_models_trained}")
        if l2_dimensions:
            for dim in l2_dimensions:
                print(f"  ✓ L2 {dim} models trained: {l2_models_trained[dim]}")

        # Explicit memory cleanup after all entities processed
        import gc
        gc.collect()
        print(f"  ✓ Memory freed after entity processing")

    cluster_models_trained = {window: 0 for window in observation_windows}

    if train_cluster:
        print(f"\n{'#'*60}")
        print("PHASE 3: Cluster Model Training (L1)")
        print(f"{'#'*60}")

        MIN_CLUSTER_SAMPLES = 100

        l1_trainer = ModelTrainer(
            input_dim=len(L1_FEATURE_ORDER),
            encoding_dim=12,
            hidden_dim=30,
            batch_size=256,
            learning_rate=0.001,
            epochs=100
        )

        for window in observation_windows:
            print(f"\n  [PHASE 3/3] Training cluster models for {window}min window...")

            cluster_models_path = models_base / 'cluster_models' / f'{window}min'

            for cluster_id, samples_list in cluster_data[window].items():
                samples = np.array(samples_list)

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
    else:
        print(f"\n{'#'*60}")
        print("PHASE 3: Cluster Model Training - SKIPPED")
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
