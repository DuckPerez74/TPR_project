import gc
import warnings
import time
from datetime import datetime

from core import OpenSearchClient
from data.metrics_fetcher import fetch_entity_l2_metrics
from training import IsolationForestTrainer
from training.utils.parallel import get_checkpoint_file, get_checkpoint_lock
from training.utils.checkpoint import save_completed_entity


def process_entity_l2_training(args):
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
    warnings.filterwarnings('ignore', message='.*Unverified HTTPS request.*')
    warnings.filterwarnings('ignore', message='.*SSL.*', category=Warning)

    entity_id, config, metrics_index, warmup_start, warmup_end, observation_windows, l2_dimensions, models_base = args

    client = OpenSearchClient.get_instance(config)

    def _timestamp():
        return datetime.now().strftime("%H:%M:%S")

    def _duration(start_time):
        elapsed = time.time() - start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            return f"{elapsed/60:.1f}min"
        else:
            return f"{elapsed/3600:.1f}h"

    entity_start = time.time()
    print(f"    [{_timestamp()}] [Entity {entity_id}] Starting L2 training (dimensions: {l2_dimensions}, windows: {observation_windows})")

    results = {
        'entity_id': entity_id,
        'l2_models_trained': {dim: {} for dim in l2_dimensions}
    }

    for window in observation_windows:
        for dim in l2_dimensions:
            results['l2_models_trained'][dim][window] = 0

        if 'user' in l2_dimensions:
            t2 = time.time()
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

                    if total_users > 200 and (idx % 100 == 0 or idx == total_users):
                        progress_pct = (idx / total_users) * 100
                        print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min:   -> User models: {idx}/{total_users} ({progress_pct:.0f}%, {trained_count} trained)")

                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: ✓ {results['l2_models_trained']['user'][window]} user models trained in {_duration(t3)}")

        if 'route' in l2_dimensions:
            t4 = time.time()
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
                            trainer.save_model(model_data, route_value, route_models_path, entity_id=entity_id)
                            results['l2_models_trained']['route'][window] += 1
                            trained_count += 1

                    if total_routes > 200 and (idx % 100 == 0 or idx == total_routes):
                        progress_pct = (idx / total_routes) * 100
                        print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min:   -> Route models: {idx}/{total_routes} ({progress_pct:.0f}%, {trained_count} trained)")

                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: ✓ {results['l2_models_trained']['route'][window]} route models trained in {_duration(t5)}")

    total_time = _duration(entity_start)
    total_user = sum(results['l2_models_trained'].get('user', {}).values())
    total_route = sum(results['l2_models_trained'].get('route', {}).values())

    print(f"    [{_timestamp()}] [Entity {entity_id}] ✓ Completed all windows in {total_time} (User:{total_user}, Route:{total_route})")

    checkpoint_file = get_checkpoint_file()
    checkpoint_lock = get_checkpoint_lock()
    if checkpoint_file:
        save_completed_entity(checkpoint_file, entity_id, checkpoint_lock)

    gc.collect()

    return results
