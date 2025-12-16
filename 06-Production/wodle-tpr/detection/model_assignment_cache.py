import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple
import sys


class ModelAssignment:
    """Represents a model assignment for an entity."""

    def __init__(self, entity_id: str, assigned_model: str, model_type: str,
                 cluster_id: Optional[int] = None, assigned_at: Optional[datetime] = None,
                 expires_at: Optional[datetime] = None, confidence: Optional[float] = None,
                 model_exists: bool = True):
        self.entity_id = entity_id
        self.assigned_model = assigned_model
        self.model_type = model_type  # 'entity' or 'cluster'
        self.cluster_id = cluster_id
        self.assigned_at = assigned_at or datetime.utcnow()
        self.expires_at = expires_at
        self.confidence = confidence
        self.model_exists = model_exists

    def is_expired(self) -> bool:
        """Check if assignment has expired."""
        if self.expires_at is None:
            return False  # Never expires (entity models)
        return datetime.utcnow() > self.expires_at

    def __repr__(self):
        expires_str = self.expires_at.isoformat() if self.expires_at else 'never'
        return (f"ModelAssignment(entity={self.entity_id}, model={self.assigned_model}, "
                f"type={self.model_type}, cluster={self.cluster_id}, expires={expires_str})")


class ModelAssignmentCache:
    """
    Persistent cache for entity-to-model assignments.

    This cache stores which model (entity-specific or cluster) should be used for each entity,
    avoiding repeated cluster predictions and WindowBuffer queries.

    Entity models never expire. Cluster assignments expire after cluster_ttl_days.
    """

    def __init__(self, db_path: str, cluster_ttl_days: int = 7):
        """
        Initialize the cache.

        Args:
            db_path: Path to SQLite database file
            cluster_ttl_days: Days until cluster assignments expire (default: 7)
        """
        self.db_path = Path(db_path)
        self.cluster_ttl_days = cluster_ttl_days
        self._init_db()

        print(f"[ModelAssignmentCache] Initialized at {self.db_path} "
              f"(cluster_ttl={cluster_ttl_days} days)",
              file=sys.stderr)

    def _init_db(self):
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_assignments (
                    entity_id TEXT PRIMARY KEY,
                    assigned_model TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    cluster_id INTEGER,
                    assigned_at TEXT NOT NULL,
                    expires_at TEXT,
                    confidence REAL,
                    hit_count INTEGER DEFAULT 0,
                    last_hit TEXT,
                    model_exists INTEGER DEFAULT 1
                )
            """)

            # Add model_exists column if it doesn't exist (for existing DBs)
            try:
                conn.execute("""
                    ALTER TABLE model_assignments
                    ADD COLUMN model_exists INTEGER DEFAULT 1
                """)
            except sqlite3.OperationalError:
                # Column already exists
                pass

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires
                ON model_assignments(expires_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_type
                ON model_assignments(model_type)
            """)

            conn.commit()

    def get(self, entity_id: str) -> Optional[ModelAssignment]:
        """
        Get model assignment for entity.

        Args:
            entity_id: Entity identifier

        Returns:
            ModelAssignment if found and not expired, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT assigned_model, model_type, cluster_id,
                       assigned_at, expires_at, confidence, hit_count, model_exists
                FROM model_assignments
                WHERE entity_id = ?
            """, (entity_id,))

            row = cursor.fetchone()

            if not row:
                return None

            assignment = ModelAssignment(
                entity_id=entity_id,
                assigned_model=row[0],
                model_type=row[1],
                cluster_id=row[2],
                assigned_at=datetime.fromisoformat(row[3]) if row[3] else None,
                expires_at=datetime.fromisoformat(row[4]) if row[4] else None,
                confidence=row[5],
                model_exists=bool(row[7]) if len(row) > 7 else True
            )

            # Check if expired
            if assignment.is_expired():
                self.delete(entity_id)
                return None

            # Update hit count and last hit time
            conn.execute("""
                UPDATE model_assignments
                SET hit_count = hit_count + 1,
                    last_hit = ?
                WHERE entity_id = ?
            """, (datetime.utcnow().isoformat(), entity_id))
            conn.commit()

            return assignment

    def set_entity_model(self, entity_id: str, model_name: str, model_exists: bool = True):
        """
        Assign entity-specific model (never expires).

        Args:
            entity_id: Entity identifier
            model_name: Model name (e.g., 'entity_125')
            model_exists: Whether the model file exists on disk
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_assignments
                (entity_id, assigned_model, model_type, cluster_id,
                 assigned_at, expires_at, confidence, hit_count, last_hit, model_exists)
                VALUES (?, ?, 'entity', NULL, ?, NULL, NULL, 0, NULL, ?)
            """, (entity_id, model_name, datetime.utcnow().isoformat(), int(model_exists)))
            conn.commit()

        exists_str = "EXISTS" if model_exists else "MISSING"
        print(f"[ModelAssignmentCache] Assigned entity model: {entity_id} -> {model_name} [{exists_str}]",
              file=sys.stderr)

    def set_cluster_model(self, entity_id: str, cluster_id: int, confidence: Optional[float] = None,
                          model_exists: bool = True):
        """
        Assign cluster model (expires in cluster_ttl_days).

        Args:
            entity_id: Entity identifier
            cluster_id: Cluster ID (0, 1, or 2)
            confidence: Optional prediction confidence score
            model_exists: Whether the model file exists on disk
        """
        model_name = f"cluster_{cluster_id}"
        expires_at = datetime.utcnow() + timedelta(days=self.cluster_ttl_days)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_assignments
                (entity_id, assigned_model, model_type, cluster_id,
                 assigned_at, expires_at, confidence, hit_count, last_hit, model_exists)
                VALUES (?, ?, 'cluster', ?, ?, ?, ?, 0, NULL, ?)
            """, (entity_id, model_name, cluster_id,
                  datetime.utcnow().isoformat(),
                  expires_at.isoformat(), confidence, int(model_exists)))
            conn.commit()

        exists_str = "EXISTS" if model_exists else "MISSING"
        print(f"[ModelAssignmentCache] Assigned cluster model: {entity_id} -> cluster_{cluster_id} "
              f"[{exists_str}] (expires: {expires_at.strftime('%Y-%m-%d %H:%M')})",
              file=sys.stderr)

    def delete(self, entity_id: str):
        """
        Remove assignment for entity.

        Args:
            entity_id: Entity identifier
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM model_assignments WHERE entity_id = ?", (entity_id,))
            conn.commit()

    def clear_expired(self) -> int:
        """
        Remove all expired assignments.

        Returns:
            Number of assignments deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM model_assignments
                WHERE expires_at IS NOT NULL
                AND expires_at < ?
            """, (datetime.utcnow().isoformat(),))
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            print(f"[ModelAssignmentCache] Cleared {deleted} expired assignments",
                  file=sys.stderr)

        return deleted

    def clear_all(self):
        """Clear all assignments (useful for re-training)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM model_assignments")
            deleted = cursor.rowcount
            conn.commit()

        print(f"[ModelAssignmentCache] Cleared all {deleted} assignments",
              file=sys.stderr)

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN model_type = 'entity' THEN 1 ELSE 0 END) as entities,
                    SUM(CASE WHEN model_type = 'cluster' THEN 1 ELSE 0 END) as clusters,
                    SUM(hit_count) as total_hits,
                    AVG(CASE WHEN model_type = 'cluster' THEN confidence END) as avg_confidence
                FROM model_assignments
            """)
            row = cursor.fetchone()

            cursor_expired = conn.execute("""
                SELECT COUNT(*)
                FROM model_assignments
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (datetime.utcnow().isoformat(),))
            expired_count = cursor_expired.fetchone()[0]

            return {
                'total_assignments': row[0] or 0,
                'entity_models': row[1] or 0,
                'cluster_models': row[2] or 0,
                'total_cache_hits': row[3] or 0,
                'avg_cluster_confidence': float(row[4]) if row[4] else 0.0,
                'expired_assignments': expired_count,
                'cluster_ttl_days': self.cluster_ttl_days
            }

    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        Get distribution of entities across clusters.

        Returns:
            Dictionary mapping cluster_id to count
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT cluster_id, COUNT(*)
                FROM model_assignments
                WHERE model_type = 'cluster'
                  AND (expires_at IS NULL OR expires_at > ?)
                GROUP BY cluster_id
            """, (datetime.utcnow().isoformat(),))

            return {int(row[0]): row[1] for row in cursor.fetchall() if row[0] is not None}

    def get_top_entities(self, limit: int = 10) -> list:
        """
        Get entities with most cache hits.

        Args:
            limit: Maximum number of entities to return

        Returns:
            List of tuples (entity_id, assigned_model, hit_count)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT entity_id, assigned_model, hit_count
                FROM model_assignments
                ORDER BY hit_count DESC
                LIMIT ?
            """, (limit,))

            return cursor.fetchall()

    def sync_from_disk(self, model_loader):
        """
        Synchronize cache with models that exist on disk.

        Args:
            model_loader: ModelLoader instance to check which models exist
        """
        from pathlib import Path

        added_count = 0
        updated_count = 0

        # Sync entity models
        if model_loader.entity_models_path.exists():
            for model_file in model_loader.entity_models_path.glob("entity_*.pt"):
                entity_id = model_file.stem.replace("entity_", "")
                model_name = f"entity_{entity_id}"

                # Check if already in cache
                existing = self.get(entity_id)
                if existing is None:
                    self.set_entity_model(entity_id, model_name, model_exists=True)
                    added_count += 1
                elif existing.model_type == 'cluster':
                    # Entity now has own model, upgrade from cluster
                    self.set_entity_model(entity_id, model_name, model_exists=True)
                    updated_count += 1

        # Sync cluster models (for entities not having entity models)
        # This would require knowing which entities should use which cluster
        # Skip for now as cluster assignment requires KMeans prediction

        print(f"[ModelAssignmentCache] Sync complete: {added_count} added, {updated_count} updated",
              file=sys.stderr)

        return added_count, updated_count
