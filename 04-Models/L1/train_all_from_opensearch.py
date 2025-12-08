import argparse
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
from opensearchpy import OpenSearch

ES_HOST = "https://100.125.228.80:9200"
ES_USER = "admin"
ES_PASSWORD = "SecretPassword"


def get_es_client():
    """Create OpenSearch client."""
    client = OpenSearch(
        hosts=[ES_HOST],
        http_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=60,
    )
    if not client.ping():
        raise ConnectionError("Failed to connect to OpenSearch")
    return client


def fetch_all_entities(window, start_date, end_date):
    print("="*70)
    print("FETCHING ENTITIES FROM OPENSEARCH")
    print("="*70)
    print(f"Index: metrics-l1-{window}m*")
    print(f"Period: {start_date} to {end_date}")
    print()

    client = get_es_client()
    index_pattern = f"metrics-l1-{window}m*"

    query = {
        "size": 0,
        "query": {
            "range": {
                "@timestamp": {
                    "gte": start_date,
                    "lt": end_date
                }
            }
        },
        "aggs": {
            "unique_companies": {
                "terms": {
                    "field": "company_id.keyword",
                    "size": 10000,
                    "order": {"_count": "desc"}
                }
            }
        }
    }

    print(f"Querying {index_pattern}...")
    response = client.search(index=index_pattern, body=query)

    buckets = response['aggregations']['unique_companies']['buckets']
    entities = [bucket['key'] for bucket in buckets]

    print(f"[OK] Found {len(entities)} unique entities")
    print()

    if len(entities) > 0:
        print(f"Sample entities: {entities[:10]}")
        if len(entities) > 10:
            print(f"... and {len(entities)-10} more")

    print("="*70)
    print()

    return entities


def check_model_exists(entity_id, window, base_output_dir):
    """Check if model already exists."""
    entity_dir = Path(base_output_dir) / f"entity-{entity_id}"
    model_name = f"entity-{entity_id}-{window}m.pth"
    model_path = entity_dir / model_name
    return model_path.exists()


def train_entity_model(entity_id, window, train_start, train_end, test_start, test_end, base_output_dir, device):
    """Train model for a single entity."""

    entity_dir = Path(base_output_dir) / f"entity-{entity_id}"
    os.makedirs(entity_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "train_autoencoder.py",
        "--mode", "entity",
        "--entity-id", str(entity_id),
        "--window", str(window),
        "--train-start", train_start,
        "--train-end", train_end,
        "--output-dir", str(entity_dir),
        "--device", device
    ]

    if test_start and test_end:
        cmd.extend(["--test-start", test_start, "--test-end", test_end])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode == 0:
            return {"status": "success", "message": "Model trained successfully"}
        else:
            stderr_lines = result.stderr.split('\n')
            error_msg = "Unknown error"
            for line in reversed(stderr_lines):
                if line.strip() and not line.startswith('  '):
                    error_msg = line.strip()
                    break
            return {"status": "failed", "message": error_msg}

    except subprocess.TimeoutExpired:
        return {"status": "failed", "message": "Training timeout (>1 hour)"}
    except Exception as e:
        return {"status": "failed", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description='Train models for all entities from OpenSearch')
    parser.add_argument('--window', type=int, required=True, choices=[10, 30, 60],
                        help='Time window in minutes')
    parser.add_argument('--train-start', type=str, required=True,
                        help='Training start date (ISO 8601)')
    parser.add_argument('--train-end', type=str, required=True,
                        help='Training end date (ISO 8601)')
    parser.add_argument('--test-start', type=str, default=None,
                        help='Test start date (ISO 8601)')
    parser.add_argument('--test-end', type=str, default=None,
                        help='Test end date (ISO 8601)')
    parser.add_argument('--output-dir', type=str, default='out',
                        help='Base output directory (creates entity-{ID} subdirs, default: ./out)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, or cpu (default: auto)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip entities with existing models')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of entities to train (for testing)')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from entity index (for resuming)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Skip entities with less than N samples (default: 100)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        entity_ids = fetch_all_entities(args.window, args.train_start, args.train_end)
    except Exception as e:
        print(f"[ERROR] Failed to fetch entities from OpenSearch: {e}")
        return 1

    if not entity_ids:
        print("[ERROR] No entities found in OpenSearch for the specified period")
        return 1

    total_entities = len(entity_ids)

    if args.limit:
        entity_ids = entity_ids[:args.limit]
        print(f"[INFO] Limiting to first {args.limit} entities (testing mode)")
        print()

    if args.start_from > 0:
        entity_ids = entity_ids[args.start_from:]
        print(f"[INFO] Resuming from entity index {args.start_from}")
        print()

    print("="*70)
    print("BATCH TRAINING - ALL ENTITIES FROM OPENSEARCH")
    print("="*70)
    print(f"Window: {args.window}m")
    print(f"Train period: {args.train_start} to {args.train_end}")
    print(f"Output: {args.output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Total entities to process: {len(entity_ids)}/{total_entities}")
    print("="*70)
    print()

    results = {
        'skipped': [],
        'success': [],
        'failed': [],
        'insufficient_data': []
    }

    start_time = datetime.now()

    for idx, entity_id in enumerate(entity_ids, 1):
        print(f"\n[{idx}/{len(entity_ids)}] Processing entity {entity_id}...")
        print("-" * 70)

        if args.skip_existing and check_model_exists(entity_id, args.window, args.output_dir):
            print(f"[SKIP] Model already exists: entity-{entity_id}-{args.window}m.pth")
            results['skipped'].append(entity_id)
            continue

        print(f"Training entity-{entity_id}-{args.window}m...")
        result = train_entity_model(
            entity_id=entity_id,
            window=args.window,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            base_output_dir=args.output_dir,
            device=args.device
        )

        if result['status'] == 'success':
            print(f"[OK] Entity {entity_id} trained successfully")
            results['success'].append(entity_id)
        else:
            error_msg = result['message']
            if 'No documents found' in error_msg or 'Insufficient' in error_msg:
                print(f"[SKIP] Entity {entity_id} - insufficient data")
                results['insufficient_data'].append({'entity_id': entity_id, 'error': error_msg})
            else:
                print(f"[ERROR] Entity {entity_id} failed: {error_msg}")
                results['failed'].append({'entity_id': entity_id, 'error': error_msg})

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("BATCH TRAINING SUMMARY")
    print("="*70)
    print(f"Total entities: {len(entity_ids)}")
    print(f"  Skipped (already exists): {len(results['skipped'])}")
    print(f"  Success: {len(results['success'])}")
    print(f"  Insufficient data: {len(results['insufficient_data'])}")
    print(f"  Failed (other errors): {len(results['failed'])}")
    print(f"Duration: {duration}")
    print("="*70)

    if results['failed']:
        print("\nFailed entities (errors):")
        for item in results['failed'][:10]:
            print(f"  - Entity {item['entity_id']}: {item['error']}")
        if len(results['failed']) > 10:
            print(f"  ... and {len(results['failed'])-10} more")

    if results['insufficient_data']:
        print(f"\nInsufficient data ({len(results['insufficient_data'])} entities):")
        sample = results['insufficient_data'][:10]
        for item in sample:
            print(f"  - Entity {item['entity_id']}")
        if len(results['insufficient_data']) > 10:
            print(f"  ... and {len(results['insufficient_data'])-10} more")

    if results['success']:
        print(f"\nSuccessful entities ({len(results['success'])} total):")
        if len(results['success']) <= 20:
            print(f"  {results['success']}")
        else:
            print(f"  {results['success'][:20]} ... and {len(results['success'])-20} more")

    report_file = Path(args.output_dir) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write("BATCH TRAINING REPORT - FROM OPENSEARCH\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Window: {args.window}m\n")
        f.write(f"Train period: {args.train_start} to {args.train_end}\n")
        f.write(f"Total entities: {len(entity_ids)}\n")
        f.write(f"Success: {len(results['success'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n")
        f.write(f"Insufficient data: {len(results['insufficient_data'])}\n")
        f.write(f"Skipped: {len(results['skipped'])}\n")
        f.write(f"Duration: {duration}\n")
        f.write("\n")

        if results['success']:
            f.write("SUCCESSFUL:\n")
            for eid in results['success']:
                f.write(f"  - entity-{eid}-{args.window}m\n")
            f.write("\n")

        if results['failed']:
            f.write("FAILED (ERRORS):\n")
            for item in results['failed']:
                f.write(f"  - Entity {item['entity_id']}: {item['error']}\n")
            f.write("\n")

        if results['insufficient_data']:
            f.write("INSUFFICIENT DATA:\n")
            for item in results['insufficient_data']:
                f.write(f"  - Entity {item['entity_id']}: {item['error']}\n")
            f.write("\n")

        if results['skipped']:
            f.write("SKIPPED (ALREADY EXISTS):\n")
            for eid in results['skipped']:
                f.write(f"  - entity-{eid}-{args.window}m\n")

    print(f"\n[OK] Detailed report saved to: {report_file}")

    return 0 if not results['failed'] else 1


if __name__ == "__main__":
    sys.exit(main())
