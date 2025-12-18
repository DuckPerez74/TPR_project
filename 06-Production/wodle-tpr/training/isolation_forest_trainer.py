import numpy as np
import joblib
import json
import re
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationForestTrainer:
    def __init__(self, n_estimators=100, contamination='auto', random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        # Updated pattern to allow route characters: /, {, }, :, ?, *, and other common URL/route chars
        self.user_id_pattern = re.compile(r'^[a-zA-Z0-9_\-\.@/{}:?*+=&%]+$')
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

            if len(X) < 50:
                return None

            if X.ndim != 2:
                import sys
                print(f"ERROR: metrics_data must be 2D array for user {user_id}", file=sys.stderr)
                return None

            # Validate data: check for NaN, Inf, or invalid values
            if not np.all(np.isfinite(X)):
                import sys
                nan_count = np.sum(np.isnan(X))
                inf_count = np.sum(np.isinf(X))
                print(f"WARNING: User/Route {user_id} has invalid values (NaN: {nan_count}, Inf: {inf_count}), skipping", file=sys.stderr)
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

    def save_model(self, model_data, user_id, output_dir, entity_id=None):
        """
        Save Isolation Forest model to disk.
        
        Args:
            model_data: Dict with 'model' and 'scaler'
            user_id: User/Route identifier
            output_dir: Output directory path
            entity_id: Optional entity_id for route models (creates entity_route naming)
        """
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
            
            # If entity_id is provided, prefix the filename with entity_id
            if entity_id:
                safe_entity_id = "".join([c if c.isalnum() else "_" for c in str(entity_id)])
                model_file = output_path / f"{safe_entity_id}_{safe_user_id}.joblib"
                scaler_file = output_path / f"{safe_entity_id}_{safe_user_id}_scaler.joblib"
            else:
                model_file = output_path / f"user_{safe_user_id}.joblib"
                scaler_file = output_path / f"user_{safe_user_id}_scaler.joblib"

            joblib.dump(model_data['model'], model_file)
            joblib.dump(model_data['scaler'], scaler_file)

            return True
        except (OSError, IOError, KeyError) as e:
            import sys
            print(f"ERROR: Failed to save model for user {user_id}: {str(e)}", file=sys.stderr)
            return False
