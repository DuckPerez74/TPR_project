import pandas as pd
import re
from .base import MetricsCalculator
from utils.metrics_helpers import calculate_entropy
from utils.geo_helpers import GeoIPHelper, haversine_distance

class L2MetricsCalculator(MetricsCalculator):
    def __init__(self, config: dict, client=None):
        super().__init__(config)
        self.client = client
        l2_config = config.get('metrics', {}).get('layers', {}).get('L2', {})
        self.dimensions = l2_config.get('dimensions', ['user', 'route'])
        self.max_dimension_value_length = 512
        
        # Initialize GeoIP Helper
        self.geo_helper = GeoIPHelper()
        
        # Cache for baselines to avoid repeated queries (Key: "dim_value_date")
        self.baseline_cache = {}

        self.dimension_field_mapping = {
            'user': 'data.operator_or_user_id',
            'route': 'data.route_uri'
        }

    def calculate(self, df: pd.DataFrame, dimension: str) -> list:
        if df.empty or dimension not in self.dimensions:
            return []

        field = self.dimension_field_mapping.get(dimension)
        if not field or field not in df.columns:
            return []

        results = []
        dimension_values = df[field].dropna().unique()

        for value in dimension_values:
            value_str = str(value)

            subset = df[df[field] == value]

            try:
                metrics = self._calculate_dimension_metrics(subset, dimension)
                results.append({
                    'dimension': dimension,
                    'dimension_value': value_str,
                    'sample_count': len(subset),
                    'metrics': metrics
                })
            except (ValueError, TypeError, KeyError) as e:
                import sys
                print(f"WARNING: L2 metrics calculation failed for {dimension}={value_str}: {str(e)}", file=sys.stderr)
                continue

        return results

    def _calculate_dimension_metrics(self, df: pd.DataFrame, dimension: str) -> dict:
        metrics = {}

        metrics['request_count'] = len(df)

        if 'data.status_code' in df.columns:
            status = pd.to_numeric(df['data.status_code'], errors='coerce').dropna()
            if not status.empty:
                total = len(status)
                if total > 0:
                    metrics['error_rate'] = round(((status >= 400).sum() / total) * 100, 2)
                    metrics['success_rate'] = round(((status < 400).sum() / total) * 100, 2)
                    # 8. Auth Failure Ratio (401, 403)
                    forbidden_count = status.isin([401, 403]).sum()
                    metrics['auth_failure_ratio'] = round(forbidden_count / total, 4)
                else:
                    metrics['error_rate'] = 0.0
                    metrics['success_rate'] = 0.0
                    metrics['auth_failure_ratio'] = 0.0

        if 'data.response_time' in df.columns:
            times = pd.to_numeric(df['data.response_time'], errors='coerce').dropna()
            if not times.empty:
                metrics['mean_response_time'] = round(times.mean(), 4)
                metrics['p95_response_time'] = round(times.quantile(0.95), 4)

        if 'data.size' in df.columns:
            sizes = pd.to_numeric(df['data.size'], errors='coerce').dropna()
            if not sizes.empty:
                metrics['mean_response_size'] = round(sizes.mean(), 2)
            else:
                metrics['mean_response_size'] = 0.0

        # --- Historical Deviations (Z-Scores) ---
        # Only calculated if we have an OpenSearch client and valid dimension value
        if self.client and dimension in ['user', 'source_ip']:
            # We need a timestamp to know "when" we are. We can pick max from df.
            # Assuming df is from the current window.
            if '@timestamp' in df.columns:
                current_ts = df['@timestamp'].max()
                # Get Dimension Value (Assuming uniform in this subset)
                dim_val_series = df[self.dimension_field_mapping[dimension]].dropna()
                if not dim_val_series.empty:
                    dim_value = str(dim_val_series.iloc[0])
                    
                    baselines = self._get_baseline(dimension, dim_value, current_ts)
                    
                    # A. Volume Deviation (Request Count)
                    curr_req = metrics.get('request_count', 0)
                    metrics['deviation_from_personal_baseline'] = 0.0
                    if baselines['req_std'] > 0:
                        metrics['deviation_from_personal_baseline'] = (curr_req - baselines['req_avg']) / baselines['req_std']

                    # B. Data Download Spike (Response Size)
                    curr_size = metrics.get('mean_response_size', 0)
                    metrics['data_download_spike'] = 0.0
                    if baselines['size_std'] > 0:
                        metrics['data_download_spike'] = (curr_size - baselines['size_avg']) / baselines['size_std']

                    # C. Error Rate Deviation
                    curr_err = metrics.get('error_rate', 0)
                    metrics['error_rate_deviation'] = 0.0
                    if baselines['err_std'] > 0:
                        metrics['error_rate_deviation'] = (curr_err - baselines['err_avg']) / baselines['err_std']
                        
                    # Add stats for debug/reference
                    metrics['hist_avg_requests'] = baselines['req_avg']
                    metrics['hist_avg_size'] = baselines['size_avg']

        if dimension == 'user':
            metrics.update(self._calculate_user_specific(df))
        elif dimension == 'source_ip':
            metrics.update(self._calculate_ip_specific(df))
        elif dimension == 'route':
            metrics.update(self._calculate_route_specific(df))

        return metrics

    def _calculate_user_specific(self, df: pd.DataFrame) -> dict:
        metrics = {}
        total = len(df)
        if total == 0: return metrics

        # Ensure string columns
        urls = df['data.route_uri'].astype(str).fillna("") if 'data.route_uri' in df.columns else pd.Series([""] * total)
        methods = df['data.method'].astype(str).fillna("") if 'data.method' in df.columns else pd.Series([""] * total)
        
        # 1. Privilege Endpoint Ratio
        admin_keywords = 'admin|config|setup|root|dashboard'
        metrics['privilege_endpoint_ratio'] = urls.str.contains(admin_keywords, case=False).sum() / total

        # 2. Sensitive Data Access
        sensitive_keywords = 'billing|finance|salary|payments|cpf|nif|ssn|credit_card'
        metrics['sensitive_data_access_rate'] = urls.str.contains(sensitive_keywords, case=False).sum() / total

        # 5. Config Modification Attempts
        config_mods = (methods.isin(['PUT', 'POST', 'PATCH', 'DELETE'])) & (urls.str.contains('config|setting', case=False))
        metrics['config_modification_attempts'] = config_mods.sum()

        # 6. Backup Access
        backup_keywords = 'backup|dump|archive|tar.gz|zip|sql'
        metrics['backup_access_indicator'] = urls.str.contains(backup_keywords, case=False).sum()

        # 7. Export Usage
        export_keywords = 'export|download|report|csv|xlsx'
        metrics['export_endpoint_usage'] = urls.str.contains(export_keywords, case=False).sum()

        # 9. Bulk Operation
        metrics['bulk_operation_ratio'] = urls.str.contains('batch|bulk|multi', case=False).sum()

        # 10. Working Hours Deviation
        if '@timestamp' in df.columns:
            log_hours = pd.to_datetime(df['@timestamp']).dt.hour
            metrics['working_hours_deviation'] = len(df[~log_hours.between(8, 19)]) / total

        # 13. Velocity Impossibility (User Dimension)
        metrics['velocity_impossibility'] = 0
        if 'data.srcip' in df.columns and '@timestamp' in df.columns:
            try:
                # Get usage sequence: Timestamp, IP
                history = df[['@timestamp', 'data.srcip']].dropna().sort_values('@timestamp')
                if len(history) > 1:
                    max_speed = 0
                    prev_loc = None
                    prev_time = None
                    
                    for _, row in history.iterrows():
                        ip = row['data.srcip']
                        ts = pd.to_datetime(row['@timestamp'])
                        
                        curr_loc = self.geo_helper.get_location(ip)
                        if not curr_loc: continue
                        
                        if prev_loc and prev_time:
                            dist = haversine_distance(prev_loc['lon'], prev_loc['lat'], curr_loc['lon'], curr_loc['lat'])
                            time_diff = (ts - prev_time).total_seconds() / 3600.0 # Hours
                            
                            if time_diff > 0.01 and dist > 50: # Ignore tiny jumps
                                speed = dist / time_diff
                                if speed > max_speed: max_speed = speed
                        
                        prev_loc = curr_loc
                        prev_time = ts
                    
                    if max_speed > 800:
                        metrics['velocity_impossibility'] = 1
            except Exception:
                pass

        # Existing metrics
        metrics['unique_routes_accessed'] = df['data.route_uri'].nunique() if 'data.route_uri' in df.columns else 0
        if 'data.route_uri' in df.columns:
            metrics['route_diversity'] = round(calculate_entropy(df['data.route_uri']), 4)

        if 'data.method' in df.columns:
            method_counts = df['data.method'].value_counts()
            if not method_counts.empty:
                metrics['primary_http_method'] = method_counts.index[0]

        return metrics

    def _calculate_ip_specific(self, df: pd.DataFrame) -> dict:
        metrics = {}
        total = len(df)
        if total == 0: return metrics

        # Ensure string columns
        urls = df['data.route_uri'].astype(str).fillna("") if 'data.route_uri' in df.columns else pd.Series([""] * total)
        methods = df['data.method'].astype(str).fillna("") if 'data.method' in df.columns else pd.Series([""] * total)

        # --- Behavioral Metrics (Ported for IP Dimension) ---
        
        # 1. Privilege Endpoint Ratio
        admin_keywords = 'admin|config|setup|root|dashboard'
        metrics['privilege_endpoint_ratio'] = urls.str.contains(admin_keywords, case=False).sum() / total

        # 2. Sensitive Data Access
        sensitive_keywords = 'billing|finance|salary|payments|cpf|nif|ssn|credit_card'
        metrics['sensitive_data_access_rate'] = urls.str.contains(sensitive_keywords, case=False).sum() / total

        # 5. Config Modification Attempts
        config_mods = (methods.isin(['PUT', 'POST', 'PATCH', 'DELETE'])) & (urls.str.contains('config|setting', case=False))
        metrics['config_modification_attempts'] = config_mods.sum()

        # 6. Backup Access
        backup_keywords = 'backup|dump|archive|tar.gz|zip|sql'
        metrics['backup_access_indicator'] = urls.str.contains(backup_keywords, case=False).sum()

        # 7. Export Usage
        export_keywords = 'export|download|report|csv|xlsx'
        metrics['export_endpoint_usage'] = urls.str.contains(export_keywords, case=False).sum()

        # 9. Bulk Operation
        metrics['bulk_operation_ratio'] = urls.str.contains('batch|bulk|multi', case=False).sum()

        # 10. Working Hours Deviation
        if '@timestamp' in df.columns:
            log_hours = pd.to_datetime(df['@timestamp']).dt.hour
            metrics['working_hours_deviation'] = len(df[~log_hours.between(8, 19)]) / total

        # --- Base IP Metrics ---

        if 'data.operator_or_user_id' in df.columns:
            metrics['unique_users'] = df['data.operator_or_user_id'].nunique()

        if 'data.route_uri' in df.columns:
            metrics['unique_routes'] = df['data.route_uri'].nunique()

        if 'data.browser' in df.columns:
            ua_counts = df['data.browser'].nunique()
            metrics['user_agent_switches'] = ua_counts

        return metrics

    def _calculate_route_specific(self, df: pd.DataFrame) -> dict:
        metrics = {}

        if 'data.operator_or_user_id' in df.columns:
            metrics['unique_users'] = df['data.operator_or_user_id'].nunique()

        if 'data.srcip' in df.columns:
            metrics['unique_source_ips'] = df['data.srcip'].nunique()

        if 'data.method' in df.columns:
            method_counts = df['data.method'].value_counts()
            if not method_counts.empty:
                total = method_counts.sum()
                metrics['method_distribution'] = {
                    method: round((count / total) * 100, 2)
                    for method, count in method_counts.items()
                }

        return metrics

    def _get_baseline(self, dimension: str, value: str, current_time) -> dict:
        """
        Fetch historical stats for a specific dimension value (User/IP).
        Uses a cache to avoid repeated queries for the same entity in the same day.
        """
        # Cache Key: dimension_value_date (e.g., "user_alice_2025-08-01")
        date_key = current_time.strftime("%Y-%m-%d") if hasattr(current_time, 'strftime') else str(current_time)[:10]
        cache_key = f"{dimension}_{value}_{date_key}"

        # Limit cache size to prevent memory bloat (max 1000 entries)
        if len(self.baseline_cache) > 1000:
            # Clear oldest half of cache
            keys_to_remove = list(self.baseline_cache.keys())[:500]
            for k in keys_to_remove:
                del self.baseline_cache[k]

        if cache_key in self.baseline_cache:
            return self.baseline_cache[cache_key]
            
        # Default (Neutral) Baselines
        baselines = {
            'req_avg': 0, 'req_std': 1,
            'size_avg': 0, 'size_std': 1,
            'err_avg': 0, 'err_std': 1
        }
        
        try:
            # Query historic L2 metrics for this user
            end_date = pd.to_datetime(current_time)
            start_date = end_date - pd.Timedelta(days=15)
            
            query = {
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"dimension": dimension}},
                            {"term": {"dimension_value": value}},
                            {"term": {"layer": "L2"}},
                            {
                                "range": {
                                    "@timestamp": {
                                        # Use standard ISO format
                                        "gte": start_date.isoformat(),
                                        "lt": end_date.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "aggs": {
                    "stats_requests": {"extended_stats": {"field": "metrics.request_count"}},
                    "stats_size": {"extended_stats": {"field": "metrics.mean_response_size"}},
                    "stats_errors": {"extended_stats": {"field": "metrics.error_rate"}}
                }
            }
            
            response = self.client.search(index="metrics-tpr", body=query, ignore_unavailable=True)
            
            def extract_stats(agg_name):
                stats = response.get('aggregations', {}).get(agg_name, {})
                avg = stats.get('avg')
                std = stats.get('std_deviation')
                if avg is None: avg = 0
                if std is None or std == 0: std = 1
                return avg, std

            baselines['req_avg'], baselines['req_std'] = extract_stats('stats_requests')
            baselines['size_avg'], baselines['size_std'] = extract_stats('stats_size')
            baselines['err_avg'], baselines['err_std'] = extract_stats('stats_errors')

        except Exception as e:
            # Silently fail and return defaults to avoid interrupting warmup
            pass
            
        # Save to cache
        self.baseline_cache[cache_key] = baselines
        return baselines
