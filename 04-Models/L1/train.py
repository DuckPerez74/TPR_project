import argparse
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Unified Auto-Encoder Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='mode', help='Training mode')

    single = subparsers.add_parser('single', help='Train a single entity or cluster model')
    single.add_argument('--entity-id', type=str, help='Entity ID')
    single.add_argument('--cluster-id', type=int, help='Cluster ID')
    single.add_argument('--entity-file', type=str, help='CSV file with company_ids')
    single.add_argument('--window', type=int, required=True, choices=[10, 30, 60], help='Time window (minutes)')
    single.add_argument('--train-start', type=str, required=True, help='Train start (ISO 8601)')
    single.add_argument('--train-end', type=str, required=True, help='Train end (ISO 8601)')
    single.add_argument('--test-start', type=str, help='Test start (ISO 8601)')
    single.add_argument('--test-end', type=str, help='Test end (ISO 8601)')
    single.add_argument('--output-dir', type=str, default='out', help='Output directory (default: out)')
    single.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu (default: auto)')

    all_entities = subparsers.add_parser('all-entities', help='Train all entities from OpenSearch')
    all_entities.add_argument('--window', type=int, required=True, choices=[10, 30, 60], help='Time window (minutes)')
    all_entities.add_argument('--train-start', type=str, required=True, help='Train start (ISO 8601)')
    all_entities.add_argument('--train-end', type=str, required=True, help='Train end (ISO 8601)')
    all_entities.add_argument('--test-start', type=str, help='Test start (ISO 8601)')
    all_entities.add_argument('--test-end', type=str, help='Test end (ISO 8601)')
    all_entities.add_argument('--output-dir', type=str, default='out', help='Output directory (default: out)')
    all_entities.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu (default: auto)')
    all_entities.add_argument('--skip-existing', action='store_true', help='Skip already trained models')
    all_entities.add_argument('--limit', type=int, help='Limit number of entities (for testing)')
    all_entities.add_argument('--start-from', type=int, default=0, help='Resume from entity index')

    all_clusters = subparsers.add_parser('all-clusters', help='Train all cluster models')
    all_clusters.add_argument('--clusters-dir', type=str, default='../../03-Clusters', help='Clusters directory (default: ../../03-Clusters)')
    all_clusters.add_argument('--window', type=int, required=True, choices=[10, 30, 60], help='Time window (minutes)')
    all_clusters.add_argument('--train-start', type=str, required=True, help='Train start (ISO 8601)')
    all_clusters.add_argument('--train-end', type=str, required=True, help='Train end (ISO 8601)')
    all_clusters.add_argument('--test-start', type=str, help='Test start (ISO 8601)')
    all_clusters.add_argument('--test-end', type=str, help='Test end (ISO 8601)')
    all_clusters.add_argument('--output-dir', type=str, default='out', help='Output directory (default: out)')
    all_clusters.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu (default: auto)')
    all_clusters.add_argument('--skip-existing', action='store_true', help='Skip already trained models')

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return 1

    if args.mode == 'single':
        if args.entity_id and args.cluster_id:
            print("[ERROR] Cannot specify both --entity-id and --cluster-id")
            return 1

        if not args.entity_id and not args.cluster_id:
            print("[ERROR] Must specify either --entity-id or --cluster-id")
            return 1

        if args.cluster_id is not None and not args.entity_file:
            print("[ERROR] --entity-file required for cluster mode")
            return 1

        cmd = [sys.executable, "train_autoencoder.py"]

        if args.entity_id:
            entity_dir = Path(args.output_dir) / f"entity-{args.entity_id}"
            cmd.extend([
                "--mode", "entity",
                "--entity-id", str(args.entity_id),
                "--output-dir", str(entity_dir)
            ])
        else:
            cluster_dir = Path(args.output_dir) / f"cluster-{args.cluster_id}"
            cmd.extend([
                "--mode", "cluster",
                "--cluster-id", str(args.cluster_id),
                "--entity-file", args.entity_file,
                "--output-dir", str(cluster_dir)
            ])

        cmd.extend([
            "--window", str(args.window),
            "--train-start", args.train_start,
            "--train-end", args.train_end,
            "--device", args.device
        ])

        if args.test_start and args.test_end:
            cmd.extend(["--test-start", args.test_start, "--test-end", args.test_end])

        return subprocess.call(cmd)

    elif args.mode == 'all-entities':
        cmd = [
            sys.executable, "train_all_from_opensearch.py",
            "--window", str(args.window),
            "--train-start", args.train_start,
            "--train-end", args.train_end,
            "--output-dir", args.output_dir,
            "--device", args.device
        ]

        if args.test_start and args.test_end:
            cmd.extend(["--test-start", args.test_start, "--test-end", args.test_end])

        if args.skip_existing:
            cmd.append("--skip-existing")

        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

        if args.start_from > 0:
            cmd.extend(["--start-from", str(args.start_from)])

        return subprocess.call(cmd)

    elif args.mode == 'all-clusters':
        cmd = [
            sys.executable, "train_cluster_models.py",
            "--clusters-dir", args.clusters_dir,
            "--window", str(args.window),
            "--train-start", args.train_start,
            "--train-end", args.train_end,
            "--output-dir", args.output_dir,
            "--device", args.device
        ]

        if args.test_start and args.test_end:
            cmd.extend(["--test-start", args.test_start, "--test-end", args.test_end])

        if args.skip_existing:
            cmd.append("--skip-existing")

        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
