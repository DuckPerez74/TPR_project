import pandas as pd
import re
from datetime import datetime


class DataPreprocessor:
    def __init__(self, config: dict):
        self.company_id_field = config.get('indices', {}).get('raw_logs', {}).get('company_id_field', 'data.entities')
        self.entity_id_pattern = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
        self.max_entity_id_length = 256

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        if 'timestamp' in df.columns and '@timestamp' in df.columns:
            # If both exist, drop 'timestamp' to avoid collision after rename
            df.drop(columns=['timestamp'], inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': '@timestamp'}, inplace=True)

        if '@timestamp' in df.columns:
            df['@timestamp'] = pd.to_datetime(df['@timestamp'], errors='coerce')
            df = df.dropna(subset=['@timestamp'])

        if self.company_id_field in df.columns:
            df = df[df[self.company_id_field].notna()]
            df[self.company_id_field] = df[self.company_id_field].astype(str).str.strip()
            df = df[~df[self.company_id_field].isin(['-', ''])]

        return df

    def get_active_entities(self, df: pd.DataFrame) -> list:
        if df.empty or self.company_id_field not in df.columns:
            return []

        entities = df[self.company_id_field].unique().tolist()
        return [e for e in entities if self._is_valid_entity_id(e)]

    def _is_valid_entity_id(self, entity_id: str) -> bool:
        if not isinstance(entity_id, str):
            return False
        if len(entity_id) == 0 or len(entity_id) > self.max_entity_id_length:
            return False
        if not self.entity_id_pattern.match(entity_id):
            return False
        return True

    def filter_by_entity(self, df: pd.DataFrame, entity_id: str) -> pd.DataFrame:
        if df.empty or self.company_id_field not in df.columns:
            return pd.DataFrame()

        if not self._is_valid_entity_id(entity_id):
            import sys
            print(f"WARNING: Invalid entity_id format rejected: {entity_id[:50]}", file=sys.stderr)
            return pd.DataFrame()

        return df[df[self.company_id_field] == entity_id].copy()
