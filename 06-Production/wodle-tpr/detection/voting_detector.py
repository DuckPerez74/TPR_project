"""
Voting-based anomaly detector for L2 user dimension.

When a user doesn't have their own model, this module finds similar users
(same entity + same account type) and uses their models for voting-based detection.
"""
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class VotingDetector:
    """
    Detects anomalies using voting from similar users' models.

    When a new user or user with insufficient training data doesn't have
    their own Isolation Forest model, this class finds similar users
    (same entity + same account) and aggregates their detection scores.
    """

    def __init__(self, model_loader, opensearch_client, score_normalizer):
        """
        Initialize voting detector.

        Args:
            model_loader: ModelLoader instance to access user models
            opensearch_client: OpenSearch client for historical queries
            score_normalizer: Function to normalize L2 scores (from detector)
        """
        self.model_loader = model_loader
        self.client = opensearch_client
        self.normalize_score = score_normalizer

        # Voting configuration
        self.min_voters = 2  # Minimum number of similar users needed
        self.avg_score_threshold = 0.4  # 40% normalized score threshold

    def detect_with_voting(self, entity_id: str, user_id: str, account: str,
                           metrics_vector: np.ndarray) -> Tuple[bool, float, str, None, Optional[dict]]:
        """
        Detect anomaly using voting from similar users' models.

        Args:
            entity_id: Entity ID
            user_id: User ID without own model
            account: Account type of the user
            metrics_vector: Metrics vector to evaluate

        Returns:
            Tuple (is_anomaly, score, model_used, cluster_id, voting_details)
            voting_details is a dict with 'voters' and 'scores' when voting is used
        """
        # Find similar users with models
        similar_users = self._find_similar_users_models(entity_id, user_id, account)

        if not similar_users:
            # No similar users found - skip L2 detection
            return False, 0.0, "voting_no_similar_users", None, None

        # Run detection with each similar user's model
        scores = []
        voter_ids = []

        for similar_user_id, model, scaler in similar_users:
            try:
                X_scaled = scaler.transform(metrics_vector)
                score = model.decision_function(X_scaled)[0]
                inverted_score = -score  # Invert: higher = more anomalous
                
                scores.append(inverted_score)
                voter_ids.append(similar_user_id)

            except Exception as e:
                print(f"WARNING: Scoring failed for similar user {similar_user_id}: {str(e)}", file=sys.stderr)
                continue

        if len(scores) < self.min_voters:
            # Not enough models to make a decision
            return False, 0.0, "voting_insufficient_voters", None, None

        avg_score = sum(scores) / len(scores)
        normalized_score = self.normalize_score(avg_score)

        # Use average score instead of voting
        is_anomaly = normalized_score > self.avg_score_threshold

        model_info = f"voting_{len(scores)}users"

        # Build voting details for logging
        voting_details = {
            'voters': voter_ids,
            'raw_scores': scores,
            'normalized_scores': [self.normalize_score(s) for s in scores]
        }

        # Enhanced stderr logging with voter details
        print(f"[VotingDetector] User {user_id} voting details:", file=sys.stderr)
        for voter, norm_score in zip(voter_ids[:10], voting_details['normalized_scores'][:10]):
            print(f"  - {voter}: {norm_score:.4f}", file=sys.stderr)
        if len(voter_ids) > 10:
            print(f"  ... and {len(voter_ids) - 10} more voters", file=sys.stderr)
        print(f"[VotingDetector] User {user_id} avg score: {normalized_score:.2f} "
              f"(threshold={self.avg_score_threshold}, is_anomaly={is_anomaly})",
              file=sys.stderr)

        return is_anomaly, float(normalized_score), model_info, None, voting_details

    def _find_similar_users_models(self, entity_id: str, exclude_user_id: str,
                                    account: str) -> List[Tuple[str, object, object]]:
        """
        Find models of similar users (same entity + same account).

        Args:
            entity_id: Entity ID to search within
            exclude_user_id: User ID to exclude (the current user without model)
            account: Account type to match (admin, manager, technician, etc.)

        Returns:
            List of tuples (user_id, model, scaler) for similar users with models
        """
        similar_models = []

        if self.client is None:
            return similar_models

        try:
            # Query recent L2 user metrics for this entity
            end_date = pd.Timestamp.utcnow()
            start_date = end_date - pd.Timedelta(days=7)

            query = {
                "size": 100,
                "_source": ["dimension_value", "metrics.account"],
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"entity_id.keyword": entity_id}},
                            {"term": {"layer.keyword": "L2"}},
                            {"term": {"dimension.keyword": "user"}},
                            {"term": {"metrics.account.keyword": account}},
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": start_date.isoformat(),
                                        "lt": end_date.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "collapse": {
                    "field": "dimension_value.keyword"
                }
            }

            response = self.client.search(index="metrics-tpr*", body=query, ignore_unavailable=True)
            hits = response.get('hits', {}).get('hits', [])

            for hit in hits:
                user_id = hit.get('_source', {}).get('dimension_value')
                if user_id and user_id != exclude_user_id:
                    # Check if this user has a model
                    model = self.model_loader.get_user_model(user_id)
                    scaler = self.model_loader.get_user_scaler(user_id)

                    if model is not None and scaler is not None:
                        similar_models.append((user_id, model, scaler))

            if similar_models:
                print(f"[VotingDetector] Found {len(similar_models)} similar users with models "
                      f"(entity={entity_id}, account={account})",
                      file=sys.stderr)

        except Exception as e:
            print(f"WARNING: Failed to find similar users: {str(e)}", file=sys.stderr)

        return similar_models
