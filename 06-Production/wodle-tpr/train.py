#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core import ConfigLoader, OpenSearchClient
from training import KMeansClusterer, ModelTrainer

FEATURE_ORDER = [
    'total_requests', 'mean_requests_per_minute', 'max_requests_per_minute',
    'std_requests_per_minute', 'cv_request_rate', 'peak_to_average_ratio', 'burst_score',
    'pct_2xx_responses', 'pct_3xx_responses', 'pct_4xx_responses', 'pct_5xx_responses',
    'error_rate', 'critical_error_rate', 'status_code_entropy', 'unique_status_codes',
    'mean_response_time', 'std_response_time', 'p50_response_time', 'p75_response_time',
    'p90_response_time', 'p95_response_time', 'p99_response_time',
    'pct_slow_requests', 'pct_very_slow_requests',
    'unique_source_ips', 'mean_requests_per_ip', 'max_requests_single_ip',
    'gini_ip_distribution', 'ip_concentration_top10pct',
    'unique_api_modules', 'module_entropy', 'top_module_percentage', 'module_switching_frequency',
    'unique_routes', 'route_entropy', 'top5_routes_percentage',
    'mean_response_size', 'std_response_size', 'max_response_size', 'min_response_size',
    'unique_user_agents', 'user_agent_entropy', 'bot_like_ua_percentage',
    'unique_http_methods', 'get_request_ratio', 'post_request_ratio'
]


def fetch_metrics_from_opensearch(client, index_name, window_minutes=60, min_samples=100, max_docs=1000000):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"layer": "L1"}},
                    {"term": {"observation_window": window_minutes}}
                ]
            }
        },
        "size": 10000
    }

    print(f"Fetching L1 metrics from {index_name} (window={window_minutes}min)...")

    all_docs = []
    scroll_id = None

    try:
        response = client.search(index=index_name, body=query, scroll='2m')
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']

        all_docs.extend(hits)

        while hits and len(all_docs) < max_docs:
            response = client.scroll(scroll_id=scroll_id, scroll='2m')
            hits = response['hits']['hits']
            all_docs.extend(hits)
            scroll_id = response.get('_scroll_id')

            if len(all_docs) >= max_docs:
                print(f"  WARNING: Reached max document limit ({max_docs}), stopping fetch", file=sys.stderr)
                break

    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception as e:
                print(f"WARNING: Failed to clear scroll context {scroll_id}: {str(e)}", file=sys.stderr)

    print(f"  Found {len(all_docs)} metric documents")

    entity_metrics = {}

    for doc in all_docs:
        source = doc['_source']
        entity_id = source.get('entity_id')
        metrics = source.get('metrics', {})

        if not entity_id:
            continue

        vector = []
        for feature in FEATURE_ORDER:
            vector.append(metrics.get(feature, 0))

        if entity_id not in entity_metrics:
            entity_metrics[entity_id] = []

        entity_metrics[entity_id].append(vector)

    entity_metrics_filtered = {
        eid: samples for eid, samples in entity_metrics.items()
        if len(samples) >= min_samples
    }

    print(f"  Entities with >={min_samples} samples: {len(entity_metrics_filtered)}")

    return entity_metrics_filtered


def train_models(config):
    print("=" * 60)
    print("TPR Model Training Pipeline")
    print("=" * 60)

    client = OpenSearchClient.get_instance(config)
    metrics_index = config.get('indices', {}).get('metrics', {}).get('name', 'metrics-tpr')

    entity_metrics = fetch_metrics_from_opensearch(client, metrics_index, window_minutes=60, min_samples=100)

    if not entity_metrics:
        print("ERROR: No entities with sufficient data found")
        sys.exit(1)

    print("\nStep 1: K-Means Clustering")
    print("-" * 60)

    entity_mean_metrics = {}
    for entity_id, samples in entity_metrics.items():
        entity_mean_metrics[entity_id] = np.mean(samples, axis=0)

    clusterer = KMeansClusterer(n_clusters=3, random_state=42)
    cluster_assignments = clusterer.fit(entity_mean_metrics)

    cluster_sizes = clusterer.get_cluster_sizes()
    print(f"  Cluster distribution: {cluster_sizes}")

    kmeans_path = Path(__file__).parent / 'models' / 'kmeans_clusterer.pkl'
    clusterer.save(kmeans_path)
    print(f"  Saved K-means to: {kmeans_path}")

    print("\nStep 2: Train Entity Models")
    print("-" * 60)

    trainer = ModelTrainer(
        input_dim=46,
        encoding_dim=12,
        hidden_dim=30,
        batch_size=256,
        learning_rate=0.001,
        epochs=100
    )

    entity_models_path = Path(__file__).parent / 'models' / 'entity_models'
    entity_models_trained = 0

    for entity_id, samples in entity_metrics.items():
        print(f"  Training entity {entity_id} ({len(samples)} samples)...")

        model_data = trainer.train_entity_model(entity_id, samples)

        if model_data:
            trainer.save_model(model_data, entity_id, entity_models_path)
            entity_models_trained += 1

    print(f"  Trained {entity_models_trained} entity models")

    print("\nStep 3: Train Cluster Models")
    print("-" * 60)

    cluster_data = {}
    for entity_id, samples in entity_metrics.items():
        cluster_id = cluster_assignments[entity_id]
        if cluster_id not in cluster_data:
            cluster_data[cluster_id] = []
        cluster_data[cluster_id].extend(samples)

    cluster_models_path = Path(__file__).parent / 'models' / 'cluster_models'
    cluster_models_trained = 0

    for cluster_id, samples in cluster_data.items():
        print(f"  Training cluster {cluster_id} ({len(samples)} samples)...")

        model_data = trainer.train_cluster_model(cluster_id, samples)

        if model_data:
            trainer.save_cluster_model(model_data, cluster_id, cluster_models_path)
            cluster_models_trained += 1

    print(f"  Trained {cluster_models_trained} cluster models")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Entity models: {entity_models_trained}")
    print(f"Cluster models: {cluster_models_trained}")
    print(f"K-means clusters: {len(cluster_sizes)}")
    print(f"\nModels saved to:")
    print(f"  - {entity_models_path}")
    print(f"  - {cluster_models_path}")
    print(f"  - {kmeans_path}")


def main():
    load_dotenv()

    try:
        config = ConfigLoader().get_all()
        train_models(config)
    except (KeyError, ValueError, TypeError) as e:
        print(f"ERROR: Configuration error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except (ConnectionError, TimeoutError) as e:
        print(f"ERROR: OpenSearch connection error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Training failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
