import logging
import pandas as pd
import hashlib
from datetime import datetime


class LogCompressor:
    def __init__(self):
        self.logger = logging.getLogger('wodle-tpr.llm.compressor')

    ESSENTIAL_FIELDS = {
        '@timestamp': 't',
        'data.method': 'm',
        'data.url': 'r',
        'data.status_code': 's',
        'data.response_time': 'rt',
        'data.operator_or_user_id': 'uid',
        'data.account': 'acc',
        'data.user_type': 'utype',
        'data.entities': 'eid',
        'data.srcip': 'ip',
        'data.impersonate': 'imp',
        'data.api_module': 'mod',
        'GeoLocation.country_name': 'geo_country',
        'GeoLocation.city_name': 'geo_city'
    }

    def compress_logs(self, df: pd.DataFrame) -> list:
        compressed = []

        for _, row in df.iterrows():
            log = self._compress_single_log(row)
            if log:
                compressed.append(log)

        return compressed

    def _compress_single_log(self, row: pd.Series) -> dict:
        """Compress a single log entry."""
        try:
            timestamp = row.get('@timestamp', row.get('timestamp', ''))
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except (ValueError, AttributeError):
                    time_str = str(timestamp)[-8:] if len(str(timestamp)) >= 8 else str(timestamp)
            else:
                time_str = str(timestamp)

            status = row.get('data.status_code', row.get('status_code', 0))
            try:
                status = int(status)
            except (ValueError, TypeError):
                status = 0

            response_time = row.get('data.response_time', row.get('response_time', 0))
            try:
                response_time = int(float(response_time) * 1000)
            except (ValueError, TypeError):
                response_time = 0

            geo_country = row.get('GeoLocation.country_name', row.get('geo_country', ''))
            geo_city = row.get('GeoLocation.city_name', row.get('geo_city', ''))
            geo = f"{geo_country[:2]}/{geo_city}" if geo_country and geo_city else ''

            impersonate = row.get('data.impersonate', row.get('impersonate', False))
            if isinstance(impersonate, str):
                impersonate = impersonate.lower() == 'true'

            compressed = {
                't': time_str,
                'm': str(row.get('data.method', row.get('method', ''))),
                'r': str(row.get('data.url', row.get('url', row.get('data.route_uri', '')))),
                's': status,
                'rt': response_time,
                'uid': str(row.get('data.operator_or_user_id', row.get('user_id', ''))),
                'acc': str(row.get('data.account', row.get('account', ''))),
                'utype': str(row.get('data.user_type', row.get('user_type', ''))),
                'eid': str(row.get('data.entities', row.get('entity_id', ''))),
                'ip': self._hash_ip(row.get('data.srcip', row.get('srcip', row.get('source_ip', '')))),
                'mod': str(row.get('data.api_module', row.get('api_module', '')))
            }

            if geo:
                compressed['geo'] = geo

            if impersonate:
                compressed['imp'] = True

            compressed = {k: v for k, v in compressed.items() if v and v != '0' and v != 0}

            return compressed

        except Exception as e:
            self.logger.debug(f"Failed to compress log entry: {e}")
            return None

    def _hash_ip(self, ip: str) -> str:
        """Hash IP address for privacy using SHA256. Returns format ip_xxxx."""
        if not ip:
            return ''
        ip_str = str(ip)
        hash_hex = hashlib.sha256(ip_str.encode()).hexdigest()[:4]
        return f"ip_{hash_hex}"

    def get_logs_summary(self, compressed_logs: list) -> dict:
        if not compressed_logs:
            return {}

        total = len(compressed_logs)
        methods = {}
        status_groups = {'2xx': 0, '3xx': 0, '4xx': 0, '5xx': 0}
        routes = {}

        for log in compressed_logs:
            method = log.get('m', 'UNKNOWN')
            methods[method] = methods.get(method, 0) + 1

            status = log.get('s', 0)
            if 200 <= status < 300:
                status_groups['2xx'] += 1
            elif 300 <= status < 400:
                status_groups['3xx'] += 1
            elif 400 <= status < 500:
                status_groups['4xx'] += 1
            elif status >= 500:
                status_groups['5xx'] += 1

            route = log.get('r', '')
            if route:
                routes[route] = routes.get(route, 0) + 1

        top_routes = sorted(routes.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total_logs': total,
            'methods': methods,
            'status_groups': status_groups,
            'top_routes': dict(top_routes)
        }
