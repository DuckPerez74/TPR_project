import numpy as np
import joblib
import json
import re
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationForestTrainer:
    def __init__(self, n_estimators=100, contamination=0.01, random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.user_id_pattern = re.compile(r'^[a-zA-Z0-9_\-\.@]+$')
        self.max_user_id_length = 256

    def train_user_model(self, user_id, metrics_data):
        if not self._is_valid_user_id(user_id):
            import sys
            print(f"ERROR: Invalid user_id format: {user_id[:50]}", file=sys.stderr)
            return None

        if not isinstance(metrics_data, (list, np.ndarray)):
            import sys
            print(f"ERROR: metrics_data must be list or array for user {user_id}", file=sys.stderr)
            return None

        try:
            X = np.array(metrics_data)

            if len(X) < 20:
                return None

            if X.ndim != 2:
                import sys
                print(f"ERROR: metrics_data must be 2D array for user {user_id}", file=sys.stderr)
                return None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )

            model.fit(X_scaled)

            return {
                'model': model,
                'scaler': scaler
            }
        except (ValueError, TypeError) as e:
            import sys
            print(f"ERROR: Failed to train user model for {user_id}: {str(e)}", file=sys.stderr)
            return None

    def _is_valid_user_id(self, user_id: str) -> bool:
        if not isinstance(user_id, str):
            return False
        if len(user_id) == 0 or len(user_id) > self.max_user_id_length:
            return False
        if not self.user_id_pattern.match(user_id):
            return False
        return True

    def save_model(self, model_data, user_id, output_dir):
        if not self._is_valid_user_id(user_id):
            import sys
            print(f"ERROR: Cannot save model, invalid user_id: {user_id[:50]}", file=sys.stderr)
            return False

        if not model_data or 'model' not in model_data or 'scaler' not in model_data:
            import sys
            print(f"ERROR: Invalid model_data for user {user_id}", file=sys.stderr)
            return False

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            safe_user_id = "".join([c if c.isalnum() else "_" for c in user_id])

            model_file = output_path / f"user_{safe_user_id}.joblib"
            scaler_file = output_path / f"user_{safe_user_id}_scaler.joblib"

            joblib.dump(model_data['model'], model_file)
            joblib.dump(model_data['scaler'], scaler_file)

            return True
        except (OSError, IOError, KeyError) as e:
            import sys
            print(f"ERROR: Failed to save model for user {user_id}: {str(e)}", file=sys.stderr)
            return False
