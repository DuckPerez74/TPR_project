from pathlib import Path


def load_completed_entities(checkpoint_file):
    completed = set()
    if checkpoint_file and Path(checkpoint_file).exists():
        try:
            with open(checkpoint_file, 'r') as f:
                for line in f:
                    entity_id = line.strip()
                    if entity_id:
                        completed.add(entity_id)
        except Exception as e:
            print(f"    WARNING: Could not read checkpoint file: {e}")
    return completed


def save_completed_entity(checkpoint_file, entity_id, lock=None):
    if checkpoint_file:
        try:
            if lock:
                lock.acquire()
            with open(checkpoint_file, 'a') as f:
                f.write(f"{entity_id}\n")
        except Exception as e:
            print(f"    WARNING: Could not save checkpoint for {entity_id}: {e}")
        finally:
            if lock:
                lock.release()


def delete_checkpoint(checkpoint_file):
    if checkpoint_file and Path(checkpoint_file).exists():
        try:
            Path(checkpoint_file).unlink()
            print(f"  ✓ Deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"    WARNING: Could not delete checkpoint file: {e}")
