import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .autoencoder import AutoEncoder

# Suppress PyTorch CUDA compatibility warnings (user is aware GPU may not be compatible)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')


class ModelTrainer:
    def __init__(self, input_dim=46, encoding_dim=12, hidden_dim=30,
                 batch_size=256, learning_rate=0.001, epochs=100):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_entity_model(self, entity_id, metrics_data):
        X = np.array(metrics_data)

        if len(X) < 100:
            return None

        # Validate data: check for NaN, Inf, or invalid values
        if not np.all(np.isfinite(X)):
            import sys
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            print(f"  WARNING: Entity {entity_id} has invalid values (NaN: {nan_count}, Inf: {inf_count}), skipping", file=sys.stderr)
            return None

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)

        # Use adaptive batch size to ensure training happens even with small datasets
        effective_batch_size = min(self.batch_size, len(X_train))

        # BatchNorm requires at least 2 samples per batch, so drop_last=True if batch size < 2
        # or if we might end up with a single-sample batch
        drop_last = (effective_batch_size < 2) or (len(X_train) % effective_batch_size == 1)

        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=drop_last)

        model = AutoEncoder(self.input_dim, self.encoding_dim, self.hidden_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler - reduces LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0

            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_reconstructed = model(X_val_tensor)
                val_loss = criterion(val_reconstructed, X_val_tensor).item()

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        thresholds = self._calculate_thresholds(model, X_val)

        return {
            'model': model,
            'scaler': scaler,
            'thresholds': thresholds
        }

    def train_cluster_model(self, cluster_id, cluster_metrics_data):
        return self.train_entity_model(f"cluster_{cluster_id}", cluster_metrics_data)

    def _calculate_thresholds(self, model, X_validation):
        model.eval()
        X_tensor = torch.FloatTensor(X_validation).to(self.device)

        with torch.no_grad():
            errors = model.reconstruction_error(X_tensor).cpu().numpy()

        return {
            'p90': float(np.percentile(errors, 90)),
            'p95': float(np.percentile(errors, 95)),
            'p99': float(np.percentile(errors, 99)),
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            '3sigma': float(np.mean(errors) + 3 * np.std(errors))
        }

    def save_model(self, model_data, entity_id, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_file = output_path / f"entity_{entity_id}.pt"
        scaler_file = output_path / f"entity_{entity_id}_scaler.pkl"
        threshold_file = output_path / f"entity_{entity_id}_thresholds.json"

        torch.save({
            'model_state_dict': model_data['model'].state_dict(),
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dim': self.hidden_dim
        }, model_file)

        import pickle
        with open(scaler_file, 'wb') as f:
            pickle.dump(model_data['scaler'], f)

        with open(threshold_file, 'w') as f:
            json.dump(model_data['thresholds'], f, indent=2)

    def save_cluster_model(self, model_data, cluster_id, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_file = output_path / f"cluster_{cluster_id}.pt"
        scaler_file = output_path / f"cluster_{cluster_id}_scaler.pkl"
        threshold_file = output_path / f"cluster_{cluster_id}_thresholds.json"

        torch.save({
            'model_state_dict': model_data['model'].state_dict(),
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dim': self.hidden_dim
        }, model_file)

        import pickle
        with open(scaler_file, 'wb') as f:
            pickle.dump(model_data['scaler'], f)

        with open(threshold_file, 'w') as f:
            json.dump(model_data['thresholds'], f, indent=2)
