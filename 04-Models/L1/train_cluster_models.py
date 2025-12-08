import argparse
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime


def find_cluster_files(clusters_dir):
    clusters_dir = Path(clusters_dir)
    cluster_files = list(clusters_dir.glob("cluster*_entities.csv"))

    clusters = []
    for file in cluster_files:
        try:
            cluster_id = int(file.stem.replace("cluster", "").replace("_entities", ""))
            clusters.append({
                'id': cluster_id,
                'file': str(file)
            })
        except ValueError:
            continue

    clusters.sort(key=lambda x: x['id'])
    return clusters


def check_model_exists(cluster_id, window, base_output_dir):
    cluster_dir = Path(base_output_dir) / f"cluster-{cluster_id}"
    model_name = f"cluster-{cluster_id}-{window}m.pth"
    model_path = cluster_dir / model_name
    return model_path.exists()


def train_cluster_model(cluster_id, entity_file, window, train_start, train_end, test_start, test_end, base_output_dir, device):
    cluster_dir = Path(base_output_dir) / f"cluster-{cluster_id}"
    os.makedirs(cluster_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "train_autoencoder.py",
        "--mode", "cluster",
        "--cluster-id", str(cluster_id),
        "--entity-file", entity_file,
        "--window", str(window),
        "--train-start", train_start,
        "--train-end", train_end,
        "--output-dir", str(cluster_dir),
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
            error_msg = result.stderr.split('\n')[-2] if result.stderr else "Unknown error"
            return {"status": "failed", "message": error_msg}

    except subprocess.TimeoutExpired:
        return {"status": "failed", "message": "Training timeout (>1 hour)"}
    except Exception as e:
        return {"status": "failed", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description='Train models for all clusters')
    parser.add_argument('--clusters-dir', type=str, default='../../03-Clusters',
                        help='Directory containing cluster CSV files (default: ../../03-Clusters)')
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
                        help='Base output directory (creates cluster-{ID} subdirs, default: ./out)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, or cpu (default: auto)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip clusters with existing models')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("BATCH TRAINING - CLUSTER MODELS")
    print("="*70)
    print(f"Clusters directory: {args.clusters_dir}")
    print(f"Window: {args.window}m")
    print(f"Train period: {args.train_start} to {args.train_end}")
    print(f"Output: {args.output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print("="*70)
    print()

    clusters = find_cluster_files(args.clusters_dir)

    if not clusters:
        print(f"[ERROR] No cluster files found in {args.clusters_dir}")
        print("Expected files: cluster0_entities.csv, cluster1_entities.csv, etc.")
        return 1

    print(f"Found {len(clusters)} clusters: {[c['id'] for c in clusters]}")
    print()

    results = {
        'skipped': [],
        'success': [],
        'failed': []
    }

    start_time = datetime.now()

    for idx, cluster in enumerate(clusters, 1):
        cluster_id = cluster['id']
        entity_file = cluster['file']

        print(f"\n[{idx}/{len(clusters)}] Processing cluster {cluster_id}...")
        print("-" * 70)
        print(f"Entity file: {entity_file}")

        if args.skip_existing and check_model_exists(cluster_id, args.window, args.output_dir):
            print(f"[SKIP] Model already exists: cluster-{cluster_id}-{args.window}m.pth")
            results['skipped'].append(cluster_id)
            continue

        print(f"Training cluster-{cluster_id}-{args.window}m...")
        result = train_cluster_model(
            cluster_id=cluster_id,
            entity_file=entity_file,
            window=args.window,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            base_output_dir=args.output_dir,
            device=args.device
        )

        if result['status'] == 'success':
            print(f"[OK] Cluster {cluster_id} trained successfully")
            results['success'].append(cluster_id)
        else:
            print(f"[ERROR] Cluster {cluster_id} failed: {result['message']}")
            results['failed'].append({'cluster_id': cluster_id, 'error': result['message']})

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("BATCH TRAINING SUMMARY")
    print("="*70)
    print(f"Total clusters: {len(clusters)}")
    print(f"  Skipped (already exists): {len(results['skipped'])}")
    print(f"  Success: {len(results['success'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"Duration: {duration}")
    print("="*70)

    if results['failed']:
        print("\nFailed clusters:")
        for item in results['failed']:
            print(f"  - Cluster {item['cluster_id']}: {item['error']}")

    if results['skipped']:
        print(f"\nSkipped clusters: {results['skipped']}")

    if results['success']:
        print(f"\nSuccessful clusters: {results['success']}")

    report_file = Path(args.output_dir) / f"cluster_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write("CLUSTER TRAINING REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Window: {args.window}m\n")
        f.write(f"Train period: {args.train_start} to {args.train_end}\n")
        f.write(f"Total clusters: {len(clusters)}\n")
        f.write(f"Success: {len(results['success'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n")
        f.write(f"Skipped: {len(results['skipped'])}\n")
        f.write(f"Duration: {duration}\n")
        f.write("\n")

        if results['success']:
            f.write("SUCCESSFUL:\n")
            for cid in results['success']:
                f.write(f"  - cluster-{cid}-{args.window}m\n")
            f.write("\n")

        if results['failed']:
            f.write("FAILED:\n")
            for item in results['failed']:
                f.write(f"  - Cluster {item['cluster_id']}: {item['error']}\n")
            f.write("\n")

        if results['skipped']:
            f.write("SKIPPED:\n")
            for cid in results['skipped']:
                f.write(f"  - cluster-{cid}-{args.window}m\n")

    print(f"\n[OK] Detailed report saved to: {report_file}")

    return 0 if not results['failed'] else 1


if __name__ == "__main__":
    sys.exit(main())
