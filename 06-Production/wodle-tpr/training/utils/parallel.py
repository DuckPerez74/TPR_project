import multiprocessing as mp


_gpu_semaphore = None
_checkpoint_file = None
_checkpoint_lock = None


def init_worker(semaphore, checkpoint_file=None, checkpoint_lock=None):
    global _gpu_semaphore, _checkpoint_file, _checkpoint_lock
    _gpu_semaphore = semaphore
    _checkpoint_file = checkpoint_file
    _checkpoint_lock = checkpoint_lock


def get_gpu_semaphore():
    return _gpu_semaphore


def get_checkpoint_file():
    return _checkpoint_file


def get_checkpoint_lock():
    return _checkpoint_lock


def get_optimal_workers(task_type='cpu'):
    cpu_count = mp.cpu_count()
    max_workers = 24

    print(f"    → Detected {cpu_count} CPUs")
    print(f"    → Using {max_workers} parallel workers (memory-optimized for training thousands of models)")

    return max_workers


def create_gpu_semaphore(concurrent_limit=6):
    import torch

    if torch.cuda.is_available():
        manager = mp.Manager()
        semaphore = manager.Semaphore(concurrent_limit)
        print(f"  ⚙  GPU detected! Using GPU semaphore (max {concurrent_limit} concurrent GPU trainings)")
        return semaphore, manager
    else:
        print(f"  ⚙  No GPU detected, using CPU for all training")
        return None, None
