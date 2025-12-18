import gc
import warnings
import time
from datetime import datetime

from constants import L1_FEATURE_ORDER
from core import OpenSearchClient
from data.metrics_fetcher import fetch_entity_l1_metrics
from training import ModelTrainer
from training.utils.parallel import get_gpu_semaphore, get_checkpoint_file, get_checkpoint_lock
from training.utils.checkpoint import save_completed_entity


def process_entity_l1_training(args):
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
    warnings.filterwarnings('ignore', message='.*Unverified HTTPS request.*')
    warnings.filterwarnings('ignore', message='.*SSL.*', category=Warning)

    entity_id, config, metrics_index, warmup_start, warmup_end, observation_windows, models_base = args

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
    print(f"    [{_timestamp()}] [Entity {entity_id}] Starting L1 training (windows: {observation_windows})")

    results = {
        'entity_id': entity_id,
        'l1_models_trained': {}
    }

    for window in observation_windows:
        results['l1_models_trained'][window] = 0

        t0 = time.time()
        samples = fetch_entity_l1_metrics(client, metrics_index, entity_id,
                                         warmup_start, warmup_end, window)

        print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Got {len(samples)} L1 samples (fetch: {_duration(t0)})")

        if len(samples) >= 100:
            t1 = time.time()
            print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Training L1 autoencoder...")

            gpu_semaphore = get_gpu_semaphore()
            use_gpu = gpu_semaphore is not None

            if use_gpu:
                gpu_semaphore.acquire()

            try:
                trainer = ModelTrainer(
                    input_dim=len(L1_FEATURE_ORDER),
                    encoding_dim=12,
                    hidden_dim=30,
                    batch_size=256,
                    learning_rate=0.001,
                    epochs=100,
                    force_cpu=not use_gpu
                )
                model_data = trainer.train_entity_model(entity_id, samples)
            finally:
                if use_gpu:
                    gpu_semaphore.release()

            if model_data:
                entity_models_path = models_base / 'entity_models' / f'{window}min'
                trainer.save_model(model_data, entity_id, entity_models_path)
                results['l1_models_trained'][window] = 1
                print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: ✓ L1 autoencoder trained in {_duration(t1)}")
        else:
            print(f"    [{_timestamp()}] [Entity {entity_id}] Window {window}min: Skipping L1 (insufficient samples: {len(samples)}/100)")

    total_time = _duration(entity_start)
    total_l1 = sum(results['l1_models_trained'].values())

    print(f"    [{_timestamp()}] [Entity {entity_id}] ✓ Completed all windows in {total_time} (L1:{total_l1})")

    checkpoint_file = get_checkpoint_file()
    checkpoint_lock = get_checkpoint_lock()
    if checkpoint_file:
        save_completed_entity(checkpoint_file, entity_id, checkpoint_lock)

    gc.collect()

    return results
