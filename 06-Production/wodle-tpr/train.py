#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from core import ConfigLoader
from training.orchestrator import train_all_models_unified


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='TPR Model Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            Examples:
              # Train everything (default)
              python train.py
            
              # Train only L1 models
              python train.py --l1-only
            
              # Train only L2 user models
              python train.py --l2-user-only
            
              # Train only L2 route models
              python train.py --l2-route-only
            
              # Train only cluster models
              python train.py --cluster-only
            
              # Train L1 and L2 users (skip routes and clusters)
              python train.py --no-l2-route --no-cluster
            
              # Train L2 only (both user and route)
              python train.py --no-l1 --no-cluster
                    '''
    )

    parser.add_argument('--l1-only', action='store_true',
                        help='Train only L1 entity models (skip L2 and cluster)')
    parser.add_argument('--l2-user-only', action='store_true',
                        help='Train only L2 user models (skip L1, routes, and cluster)')
    parser.add_argument('--l2-route-only', action='store_true',
                        help='Train only L2 route models (skip L1, users, and cluster)')
    parser.add_argument('--cluster-only', action='store_true',
                        help='Train only cluster models (skip L1 entity and L2)')

    parser.add_argument('--no-l1', action='store_true',
                        help='Skip L1 entity model training')
    parser.add_argument('--no-l2-user', action='store_true',
                        help='Skip L2 user model training')
    parser.add_argument('--no-l2-route', action='store_true',
                        help='Skip L2 route model training')
    parser.add_argument('--no-cluster', action='store_true',
                        help='Skip cluster model training')

    args = parser.parse_args()

    if args.l1_only:
        train_l1 = True
        train_l2_user = False
        train_l2_route = False
        train_cluster = False
    elif args.l2_user_only:
        train_l1 = False
        train_l2_user = True
        train_l2_route = False
        train_cluster = False
    elif args.l2_route_only:
        train_l1 = False
        train_l2_user = False
        train_l2_route = True
        train_cluster = False
    elif args.cluster_only:
        train_l1 = False
        train_l2_user = False
        train_l2_route = False
        train_cluster = True
    else:
        train_l1 = not args.no_l1
        train_l2_user = not args.no_l2_user
        train_l2_route = not args.no_l2_route
        train_cluster = not args.no_cluster

    if not any([train_l1, train_l2_user, train_l2_route, train_cluster]):
        print("ERROR: At least one model type must be enabled!")
        print("Use --help to see available options.")
        sys.exit(1)

    config_loader = ConfigLoader()
    config = config_loader.get_all()

    train_all_models_unified(
        config,
        train_l1=train_l1,
        train_l2_user=train_l2_user,
        train_l2_route=train_l2_route,
        train_cluster=train_cluster
    )


if __name__ == '__main__':
    main()
