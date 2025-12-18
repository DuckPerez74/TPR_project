import pandas as pd
from typing import Dict


class BehavioralMetrics:

    def __init__(self, config: dict):
        l2_config = config.get('metrics', {}).get('layers', {}).get('L2', {})

        # Load keywords from config (customizable per deployment)
        keywords = l2_config.get('keywords', {})
        self.admin_keywords = keywords.get('admin_endpoints', 'admin|config|setup|root|dashboard')
        self.sensitive_keywords = keywords.get('sensitive_data', 'billing|finance|salary|payments|cpf|nif|ssn|credit_card')
        self.backup_keywords = keywords.get('backup_files', 'backup|dump|archive|tar.gz|zip|sql')
        self.export_keywords = keywords.get('export_operations', 'export|download|report|csv|xlsx')
        self.bulk_keywords = keywords.get('bulk_operations', 'batch|bulk|multi')

        # Load thresholds from config
        thresholds = l2_config.get('thresholds', {})
        self.working_hours_start = thresholds.get('working_hours_start', 8)
        self.working_hours_end = thresholds.get('working_hours_end', 19)

        # Modification methods (from constants)
        from constants import MODIFICATION_METHODS
        self.modification_methods = MODIFICATION_METHODS

    def calculate(self, df: pd.DataFrame) -> Dict[str, float]:
        total = len(df)
        if total == 0:
            return {}

        metrics = {}

        # Ensure string columns for pattern matching
        urls = df['data.route_uri'].astype(str).fillna("") if 'data.route_uri' in df.columns else pd.Series([""] * total)
        methods = df['data.method'].astype(str).fillna("") if 'data.method' in df.columns else pd.Series([""] * total)

        # 1. Privilege Endpoint Access
        metrics['privilege_endpoint_ratio'] = (
            urls.str.contains(self.admin_keywords, case=False).sum() / total
        )

        # 2. Sensitive Data Access
        metrics['sensitive_data_access_rate'] = (
            urls.str.contains(self.sensitive_keywords, case=False).sum() / total
        )

        # 3. Configuration Modification Attempts
        config_mods = (
            methods.isin(self.modification_methods) &
            urls.str.contains('config|setting', case=False)
        )
        metrics['config_modification_attempts'] = config_mods.sum()

        # 4. Backup File Access
        metrics['backup_access_indicator'] = (
            urls.str.contains(self.backup_keywords, case=False).sum()
        )

        # 5. Export/Download Operations
        metrics['export_endpoint_usage'] = (
            urls.str.contains(self.export_keywords, case=False).sum()
        )

        # 6. Bulk Operations
        metrics['bulk_operation_ratio'] = (
            urls.str.contains(self.bulk_keywords, case=False).sum()
        )

        # 7. Working Hours Deviation
        if '@timestamp' in df.columns:
            log_hours = pd.to_datetime(df['@timestamp']).dt.hour
            metrics['working_hours_deviation'] = (
                len(df[~log_hours.between(self.working_hours_start, self.working_hours_end)]) / total
            )

        return metrics
