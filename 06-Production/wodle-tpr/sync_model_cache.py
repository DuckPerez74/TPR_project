#!/usr/bin/env python3
"""
Synchronize model assignment cache with models on disk.

This script scans all entity models on disk and updates the SQLite cache
to reflect which models exist.
"""

import json
import sys
from pathlib import Path
from detection.model_loader import ModelLoader
from detection.model_assignment_cache import ModelAssignmentCache


def main():
    # Load config
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model loader
    print("Loading models...")
    model_loader = ModelLoader(config)

    # Initialize cache
    cache_path = Path(__file__).parent / 'model_assignments.db'
    cluster_config = config.get('detection', {}).get('cluster_prediction', {})
    cluster_ttl = cluster_config.get('lookback_days', 7)
    assignment_cache = ModelAssignmentCache(cache_path, cluster_ttl_days=cluster_ttl)

    # Get initial stats
    print("\n=== Before Sync ===")
    stats_before = assignment_cache.get_stats()
    print(f"Total assignments: {stats_before['total_assignments']}")
    print(f"Entity models: {stats_before['entity_models']}")
    print(f"Cluster models: {stats_before['cluster_models']}")

    # Sync from disk
    print("\n=== Syncing from disk ===")
    added, updated = assignment_cache.sync_from_disk(model_loader)

    # Get final stats
    print("\n=== After Sync ===")
    stats_after = assignment_cache.get_stats()
    print(f"Total assignments: {stats_after['total_assignments']}")
    print(f"Entity models: {stats_after['entity_models']}")
    print(f"Cluster models: {stats_after['cluster_models']}")

    print(f"\n[OK] Sync complete: {added} added, {updated} updated")


if __name__ == '__main__':
    main()
