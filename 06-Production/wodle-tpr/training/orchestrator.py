import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from core import OpenSearchClient
from data.metrics_fetcher import get_unique_entities
from detection import ModelAssignmentCache

from training.cluster_trainer import train_kmeans_clustering, train_cluster_models
from training.entity_trainer import process_entity_l1_training
from training.l2_trainer import process_entity_l2_training
from training.utils.checkpoint import load_completed_entities
from training.utils.parallel import init_worker, get_optimal_workers, create_gpu_semaphore


def process_single_entity_all_layers(args):
    entity_id, config, metrics_index, warmup_start, warmup_end, \
        observation_windows, l2_dimensions, models_base, train_l1, train_l2 = args

    results = {
        'entity_id': entity_id,
        'l1_models_trained': {},
        'l2_models_trained': {dim: {} for dim in l2_dimensions}
    }

    if train_l1:
        l1_args = (entity_id, config, metrics_index, warmup_start, warmup_end,
                   observation_windows, models_base)
        l1_results = process_entity_l1_training(l1_args)
        results['l1_models_trained'] = l1_results['l1_models_trained']

    if train_l2 and l2_dimensions:
        l2_args = (entity_id, config, metrics_index, warmup_start, warmup_end,
                   observation_windows, l2_dimensions, models_base)
        l2_results = process_entity_l2_training(l2_args)
        results['l2_models_trained'] = l2_results['l2_models_trained']

    return results


def train_all_models_unified(config, train_l1=True, train_l2_user=True, train_l2_route=True, train_cluster=True):
    print("\n" + "=" * 60)
    print("TPR Unified Model Training Pipeline - Refactored")
    print("=" * 60)

    all_l2_dimensions = config.get('metrics', {}).get('layers', {}).get('L2', {}).get('dimensions', ['user', 'route'])
    l2_dimensions = []
    if train_l2_user and 'user' in all_l2_dimensions:
        l2_dimensions.append('user')
    if train_l2_route and 'route' in all_l2_dimensions:
        l2_dimensions.append('route')

    train_l2 = len(l2_dimensions) > 0

    print("\nTraining Configuration:")
    print(f"  L1 Entity Models: {'✓ ENABLED' if train_l1 else '✗ SKIPPED'}")
    print(f"  L2 User Models: {'✓ ENABLED' if train_l2_user else '✗ SKIPPED'}")
    print(f"  L2 Route Models: {'✓ ENABLED' if train_l2_route else '✗ SKIPPED'}")
    print(f"  Cluster Models: {'✓ ENABLED' if train_cluster else '✗ SKIPPED'}")

    warmup_config = config.get('warmup', {})
    start_date_str = warmup_config.get('start_date', 'N/A')
    end_date_str = warmup_config.get('end_date', 'N/A')

    print(f"\nWarmup Period:")
    print(f"  Start: {start_date_str}")
    print(f"  End: {end_date_str}")

    try:
        warmup_start = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        warmup_end = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
    except Exception as e:
        print(f"ERROR: Failed to parse warmup dates: {str(e)}")
        sys.exit(1)

    client = OpenSearchClient.get_instance(config)
    base_index_name = config.get('indices', {}).get('metrics', {}).get('name', 'metrics-tpr')
    metrics_index = f"{base_index_name}*"

    observation_windows = config.get('observation_windows', {}).get('enabled', [60, 30, 10])
    models_base = Path(__file__).parent.parent / 'models'

    print(f"\nMetrics Index Pattern: {metrics_index}")
    print(f"Observation Windows: {observation_windows}")
    print(f"L2 Dimensions to train: {l2_dimensions if l2_dimensions else 'None'}")

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
        cluster_assignments = train_kmeans_clustering(
            config, client, metrics_index, warmup_start, warmup_end, models_base
        )
    else:
        print(f"\n{'#'*60}")
        print("PHASE 1: K-Means Clustering - SKIPPED")
        print(f"{'#'*60}")

    if not train_l1 and not l2_dimensions:
        print(f"\n{'#'*60}")
        if train_cluster:
            print("PHASE 2: Entity-by-Entity Training - SKIPPED")
            print("  (Not needed - Cluster training uses direct queries in Phase 3)")
        else:
            print("PHASE 2: Entity-by-Entity Training - SKIPPED (no L1 or L2 selected)")
        print(f"{'#'*60}")

        l1_entity_models_trained = {window: 0 for window in observation_windows}
        l2_models_trained = {dim: {window: 0 for window in observation_windows} for dim in l2_dimensions}
    else:
        print(f"\n{'#'*60}")

        phase_parts = []
        if train_l1:
            phase_parts.append("L1")
        if l2_dimensions:
            phase_parts.append("L2")

        phase_desc = " + ".join(phase_parts)
        print(f"PHASE 2: Entity-by-Entity Training ({phase_desc}, All Windows)")
        print(f"{'#'*60}")

        print(f"  [PHASE 2/3] Fetching entities list from OpenSearch...")
        all_entities = get_unique_entities(client, metrics_index, warmup_start, warmup_end)

        if not all_entities:
            print("ERROR: No entities found!")
            sys.exit(1)

        print(f"  ✓ Total entities to process: {len(all_entities)}")

        checkpoint_file = models_base / 'training_checkpoint.txt'
        completed_entities = load_completed_entities(checkpoint_file)

        if completed_entities:
            original_count = len(all_entities)
            all_entities = [e for e in all_entities if e not in completed_entities]
            skipped_count = original_count - len(all_entities)
            print(f"\n  📋 CHECKPOINT: Found {skipped_count} already processed entities")
            print(f"  📋 Resuming from checkpoint: {len(all_entities)} entities remaining")
            print(f"  📋 Checkpoint file: {checkpoint_file}")
            if len(all_entities) == 0:
                print(f"  ✓ All entities already processed! Delete checkpoint to retrain.")
        else:
            print(f"\n  📋 Starting fresh (no checkpoint found)")
            print(f"  📋 Progress will be saved to: {checkpoint_file}")

        l1_entity_models_trained = {window: 0 for window in observation_windows}
        l2_models_trained = {dim: {window: 0 for window in observation_windows} for dim in l2_dimensions}

        if all_entities:
            print(f"\n  [PHASE 2/3] Processing {len(all_entities)} entities...")
            print(f"  ⚙  Using max {get_optimal_workers()} parallel workers")

            entity_jobs = []
            for entity_id in all_entities:
                job_args = (
                    entity_id, config, metrics_index, warmup_start, warmup_end,
                    observation_windows, l2_dimensions, models_base, train_l1, train_l2
                )
                entity_jobs.append(job_args)

            max_workers = get_optimal_workers()
            gpu_semaphore, manager = create_gpu_semaphore(concurrent_limit=6)
            checkpoint_lock = manager.Lock() if manager else None

            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker,
                initargs=(gpu_semaphore, str(checkpoint_file), checkpoint_lock)
            ) as executor:
                futures = {executor.submit(process_single_entity_all_layers, job): job[0] for job in entity_jobs}

                completed = 0
                total_l1_trained = 0
                total_l2_user_trained = 0
                total_l2_route_trained = 0

                for future in as_completed(futures):
                    entity_id = futures[future]
                    result = future.result()

                    for window in observation_windows:
                        l1_entity_models_trained[window] += result['l1_models_trained'].get(window, 0)
                        for dim in l2_dimensions:
                            l2_models_trained[dim][window] += result['l2_models_trained'][dim].get(window, 0)

                    entity_l1_count = sum(result['l1_models_trained'].values())
                    entity_l2_user_count = sum(result['l2_models_trained'].get('user', {}).values())
                    entity_l2_route_count = sum(result['l2_models_trained'].get('route', {}).values())

                    total_l1_trained += entity_l1_count
                    total_l2_user_trained += entity_l2_user_count
                    total_l2_route_trained += entity_l2_route_count

                    completed += 1
                    progress_pct = (completed / len(all_entities)) * 100

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
            print(f"  📋 Checkpoint saved: {checkpoint_file}")
            if train_l1:
                print(f"  ✓ L1 Entity models trained: {l1_entity_models_trained}")
            if l2_dimensions:
                for dim in l2_dimensions:
                    print(f"  ✓ L2 {dim} models trained: {l2_models_trained[dim]}")

            import gc
            gc.collect()
            print(f"  ✓ Memory freed after entity processing")

    if train_cluster:
        cluster_models_trained = train_cluster_models(
            config, client, metrics_index, warmup_start, warmup_end,
            cluster_assignments, observation_windows, models_base
        )
    else:
        print(f"\n{'#'*60}")
        print("PHASE 3: Cluster Model Training - SKIPPED")
        print(f"{'#'*60}")
        cluster_models_trained = {window: 0 for window in observation_windows}

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
