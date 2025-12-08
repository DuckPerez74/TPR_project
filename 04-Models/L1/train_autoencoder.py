import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from opensearchpy import OpenSearch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASSWORD

FEATURE_LIST = [
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

    'unique_http_methods', 'get_request_ratio', 'post_request_ratio',
]

ENCODING_DIM = 12
HIDDEN_DIM = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOP_PATIENCE = 15

OUTPUT_DIR = Path(".")


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=12, hidden_dim=30):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def reconstruction_error(self, x):
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


def get_es_client():
    client = OpenSearch(
        hosts=[OPENSEARCH_HOST],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=60,
    )
    if not client.ping():
        raise ConnectionError("Failed to connect to OpenSearch")
    return client

def load_entity_ids(filepath=None, manual_ids=None):
    if manual_ids:
        return manual_ids

    df = pd.read_csv(filepath)
    if 'company_id' not in df.columns:
        raise ValueError("CSV must have 'company_id' column")

    return df['company_id'].unique().tolist()

def fetch_metrics(client, entity_ids, start_time, end_time, index_pattern):
    print(f"Fetching metrics from {index_pattern}...")
    print(f"  Entities: {len(entity_ids)}")
    print(f"  Time range: {start_time} to {end_time}")

    query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": start_time, "lt": end_time}}},
                    {"terms": {"company_id": entity_ids}}
                ]
            }
        },
        "_source": ["@timestamp", "company_id", "metrics"]
    }

    response = client.search(index=index_pattern, body=query, size=10000, scroll='5m')
    scroll_id = response.get('_scroll_id')
    hits = response['hits']['hits']
    all_hits = hits.copy()

    with tqdm(total=response['hits']['total']['value'], desc="Fetching") as pbar:
        pbar.update(len(hits))

        while scroll_id and len(hits) > 0:
            response = client.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = response.get('_scroll_id')
            hits = response['hits']['hits']
            all_hits.extend(hits)
            pbar.update(len(hits))

    if not all_hits:
        raise ValueError("No documents found! Check entity IDs and date range.")

    data = []
    for hit in all_hits:
        source = hit['_source']
        row = {
            'timestamp': source['@timestamp'],
            'company_id': source['company_id'],
        }
        if 'metrics' in source:
            row.update(source['metrics'])
        data.append(row)

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} records for {df['company_id'].nunique()} entities")
    return df


def preprocess_data(df_train, df_test, feature_list):
    print(f"\nPreprocessing data...")

    missing_features = [f for f in feature_list if f not in df_train.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        feature_list = [f for f in feature_list if f in df_train.columns]
        print(f"Using {len(feature_list)} available features")

    X_train_raw = df_train[feature_list].copy()
    X_test_raw = df_test[feature_list].copy()

    nan_counts = X_train_raw.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"Found {nan_counts.sum()} NaNs, filling with 0...")
        X_train_raw = X_train_raw.fillna(0)
        X_test_raw = X_test_raw.fillna(0)

    X_train_raw = X_train_raw.replace([np.inf, -np.inf], 0)
    X_test_raw = X_test_raw.replace([np.inf, -np.inf], 0)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print(f"Preprocessed: train={X_train_scaled.shape}, test={X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler, feature_list


def train_model(X_train_scaled, val_split=0.2, device='cuda'):
    print(f"\nTraining model on {device}...")

    X_train, X_val = train_test_split(X_train_scaled, test_size=val_split, random_state=42)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim=input_dim, encoding_dim=ENCODING_DIM, hidden_dim=HIDDEN_DIM).to(device)

    print(f"Model architecture: {input_dim} -> {HIDDEN_DIM} -> {ENCODING_DIM} -> {HIDDEN_DIM} -> {input_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_reconstructed = model(X_val_tensor)
            val_loss = criterion(val_reconstructed, X_val_tensor).item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Train: {train_loss:.6f} - Val: {val_loss:.6f} - Best: {best_val_loss:.6f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

    return model, history, best_val_loss, X_train_tensor, X_val_tensor


def evaluate_model(model, X_train_tensor, X_val_tensor, X_test_scaled, device='cuda'):
    print("\nEvaluating model...")

    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    model.eval()
    with torch.no_grad():
        train_errors = model.reconstruction_error(X_train_tensor).cpu().numpy()
        val_errors = model.reconstruction_error(X_val_tensor).cpu().numpy()
        test_errors = model.reconstruction_error(X_test_tensor).cpu().numpy()

    print(f"Reconstruction Errors:")
    print(f"  Training: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
    print(f"  Validation: mean={val_errors.mean():.6f}, std={val_errors.std():.6f}")
    print(f"  Test: mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")

    thresholds = {
        'p90': float(np.percentile(val_errors, 90)),
        'p95': float(np.percentile(val_errors, 95)),
        'p99': float(np.percentile(val_errors, 99)),
        '3sigma': float(val_errors.mean() + 3 * val_errors.std()),
        'mean': float(val_errors.mean()),
        'std': float(val_errors.std()),
    }

    print(f"\nAnomaly Thresholds (from validation set):")
    for name, value in thresholds.items():
        if name not in ['mean', 'std']:
            pct = (test_errors > value).mean() * 100
            print(f"  {name}: {value:.6f} ({pct:.2f}% flagged on test set)")

    return thresholds, train_errors, val_errors, test_errors


def save_artifacts(model_name, model, scaler, feature_list, thresholds, history, metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving artifacts to {output_dir}/")

    torch.save(model.state_dict(), output_dir / f"{model_name}.pth")
    print(f"  [OK] {model_name}.pth")

    with open(output_dir / f"{model_name}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  [OK] {model_name}_scaler.pkl")

    with open(output_dir / f"{model_name}_features.json", 'w') as f:
        json.dump(feature_list, f, indent=2)
    print(f"  [OK] {model_name}_features.json")

    with open(output_dir / f"{model_name}_thresholds.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"  [OK] {model_name}_thresholds.json")

    with open(output_dir / f"{model_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] {model_name}_metadata.json")

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"{model_name}_training_history.csv", index=False)
    print(f"  [OK] {model_name}_training_history.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training History - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{model_name}_training_history.png", dpi=150)
    plt.close()
    print(f"  [OK] {model_name}_training_history.png")


def main():
    parser = argparse.ArgumentParser(description='Train L1 Auto-Encoder for Anomaly Detection')

    parser.add_argument('--mode', type=str, required=True, choices=['cluster', 'entity'],
                        help='Training mode: cluster (multiple entities) or entity (single entity)')

    parser.add_argument('--cluster-id', type=int, help='Cluster ID (for cluster mode)')
    parser.add_argument('--entity-id', type=str, help='Entity/Company ID (for entity mode)')
    parser.add_argument('--entity-file', type=str, help='Path to CSV with entity IDs (for cluster mode)')

    parser.add_argument('--window', type=int, required=True, choices=[10, 30, 60],
                        help='Time window in minutes (10, 30, or 60)')
    parser.add_argument('--train-start', type=str, required=True,
                        help='Training start date (ISO 8601, e.g., 2025-08-01T00:00:00)')
    parser.add_argument('--train-end', type=str, required=True,
                        help='Training end date (ISO 8601)')
    parser.add_argument('--test-start', type=str, help='Test start date (ISO 8601, optional)')
    parser.add_argument('--test-end', type=str, help='Test end date (ISO 8601, optional)')

    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for models (default: current directory)')

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (auto, cuda, or cpu)')

    args = parser.parse_args()

    if args.mode == 'cluster' and (not args.cluster_id or not args.entity_file):
        parser.error("--cluster-id and --entity-file required for cluster mode")

    if args.mode == 'entity' and not args.entity_id:
        parser.error("--entity-id required for entity mode")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*60)
    print("AUTO-ENCODER TRAINING")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Window: {args.window} minutes")
    print(f"Device: {device}")
    print(f"Training: {args.train_start} to {args.train_end}")
    if args.test_start:
        print(f"Testing: {args.test_start} to {args.test_end}")
    print("="*60)

    if args.mode == 'cluster':
        model_name = f"cluster-{args.cluster_id}-{args.window}m"
    else:
        model_name = f"entity-{args.entity_id}-{args.window}m"

    print(f"\nModel name: {model_name}")

    if args.mode == 'cluster':
        entity_ids = load_entity_ids(filepath=args.entity_file)
        print(f"Loaded {len(entity_ids)} entities from {args.entity_file}")
    else:
        entity_ids = [args.entity_id]
        print(f"Training for single entity: {args.entity_id}")

    print("\nConnecting to OpenSearch...")
    es_client = get_es_client()
    print("[OK] Connected")

    index_pattern = f"metrics-l1-{args.window}m*"
    df_train = fetch_metrics(es_client, entity_ids, args.train_start, args.train_end, index_pattern)

    if args.test_start:
        df_test = fetch_metrics(es_client, entity_ids, args.test_start, args.test_end, index_pattern)
    else:
        print("\nNo test dates provided, using 20% of training data for testing")
        df_test = df_train.sample(frac=0.2, random_state=42)
        df_train = df_train.drop(df_test.index)

    X_train_scaled, X_test_scaled, scaler, features_used = preprocess_data(df_train, df_test, FEATURE_LIST)

    model, history, best_val_loss, X_train_tensor, X_val_tensor = train_model(X_train_scaled, VALIDATION_SPLIT, device)

    thresholds, train_errors, val_errors, test_errors = evaluate_model(
        model, X_train_tensor, X_val_tensor, X_test_scaled, device
    )

    metadata = {
        'model_name': model_name,
        'mode': args.mode,
        'cluster_id' if args.mode == 'cluster' else 'entity_id': args.cluster_id if args.mode == 'cluster' else args.entity_id,
        'window_minutes': args.window,
        'train_period': {'start': args.train_start, 'end': args.train_end},
        'test_period': {'start': args.test_start, 'end': args.test_end} if args.test_start else None,
        'num_entities': len(entity_ids),
        'num_features': len(features_used),
        'architecture': {
            'input_dim': len(features_used),
            'hidden_dim': HIDDEN_DIM,
            'encoding_dim': ENCODING_DIM,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
        },
        'trained_at': datetime.now().isoformat(),
    }

    save_artifacts(model_name, model, scaler, features_used, thresholds, history, metadata, args.output_dir)

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Entities: {len(entity_ids)}")
    print(f"Features: {len(features_used)}")
    print(f"Training samples: {len(X_train_scaled):,}")
    print(f"Test samples: {len(X_test_scaled):,}")
    print(f"Epochs: {len(history['train_loss'])}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"\nRecommended threshold (p95): {thresholds['p95']:.6f}")
    print(f"  -> {(test_errors > thresholds['p95']).mean()*100:.2f}% flagged on test set")
    print("="*60)
    print("\n[SUCCESS] Training complete!")

if __name__ == "__main__":
    main()
