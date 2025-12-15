import json
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


class WazuhLogger:
    def __init__(self, config: dict):
        log_config = config.get('logging', {})

        self.log_file = Path(log_config.get('file', '/var/ossec/logs/anomaly_detection.log'))
        self.anomaly_log_file = Path(log_config.get('anomaly_file', '/var/ossec/logs/anomaly.log'))
        self.log_format = log_config.get('format', 'json')
        self.level = getattr(logging, log_config.get('level', 'INFO'))

        self.logger = logging.getLogger('anomaly_detection')
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()

        max_bytes = log_config.get('max_size_mb', 100) * 1024 * 1024
        backup_count = log_config.get('backup_count', 5)

        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setLevel(self.level)
        self.logger.addHandler(handler)

        self.anomaly_logger = logging.getLogger('anomaly_alerts')
        self.anomaly_logger.setLevel(logging.INFO)
        self.anomaly_logger.handlers.clear()
        self.anomaly_logger.propagate = False

        anomaly_handler = RotatingFileHandler(
            self.anomaly_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        anomaly_handler.setLevel(logging.INFO)
        anomaly_handler.setFormatter(logging.Formatter('%(message)s'))
        self.anomaly_logger.addHandler(anomaly_handler)

    def log_anomaly(self, entity_id: str, window: int, score: float,
                    layer: str, metrics_summary: dict):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'anomaly_detection',
            'severity': self._get_severity(score),
            'entity_id': entity_id,
            'layer': layer,
            'observation_window_minutes': window,
            'anomaly_score': round(score, 6),
            'metrics_summary': metrics_summary
        }

        if self.log_format == 'json':
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.info(str(log_entry))

    def log_detection_run(self, entities_analyzed: int, anomalies_found: int,
                          execution_time_ms: int):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'detection_run',
            'entities_analyzed': entities_analyzed,
            'anomalies_found': anomalies_found,
            'execution_time_ms': execution_time_ms
        }

        if self.log_format == 'json':
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.info(str(log_entry))

    def log_error(self, message: str, error: Exception = None):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'error',
            'message': message
        }

        if error:
            log_entry['error'] = str(error)
            log_entry['error_type'] = type(error).__name__

        if self.log_format == 'json':
            self.logger.error(json.dumps(log_entry))
        else:
            self.logger.error(str(log_entry))

    def log_anomaly_alert(self, entity_id: str, analysis_result: dict):
        selected_window = analysis_result.get('selected_window')
        window_result = analysis_result['results'][str(selected_window)]

        layer = window_result['anomaly_layer']
        score = window_result['score']

        alert = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'rule': {
                'id': '100001' if layer == 'L1' else '100002',
                'level': self._get_alert_level(layer, score),
                'description': f'TPR Anomaly Detected - Layer {layer}',
                'groups': ['tpr', 'anomaly_detection', f'layer_{layer.lower()}']
            },
            'data': {
                'tpr': {
                    'entity_id': entity_id,
                    'observation_window_minutes': selected_window,
                    'layer': layer,
                    'anomaly_score': round(score, 6),
                    'model_used': window_result.get('model_used'),
                    'cluster_id': window_result.get('cluster_id'),
                    'drill_down': {
                        'window_60': window_result if selected_window == 60 else analysis_result['results']['60'].get('has_anomaly', False),
                        'window_30': window_result if selected_window == 30 else analysis_result['results']['30'].get('has_anomaly', False),
                        'window_10': window_result if selected_window == 10 else analysis_result['results']['10'].get('has_anomaly', False)
                    }
                }
            },
            'decoder': {
                'name': 'tpr-anomaly'
            },
            'location': 'wodle-tpr'
        }

        if layer == 'L2':
            alert['data']['tpr']['l2_dimension'] = window_result.get('l2_dimension')
            alert['data']['tpr']['l2_dimension_value'] = window_result.get('l2_dimension_value')
            alert['data']['tpr']['l2_all_anomalies'] = window_result.get('l2_details', [])

            if window_result.get('l2_dimension') == 'user':
                alert['rule']['description'] = f'TPR User Anomaly Detected - {window_result.get("l2_dimension_value")}'
            elif window_result.get('l2_dimension') == 'route':
                alert['rule']['description'] = f'TPR Route Anomaly Detected - {window_result.get("l2_dimension_value")}'

        self.anomaly_logger.info(json.dumps(alert))

        self.logger.info(json.dumps({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'anomaly_alert_logged',
            'entity_id': entity_id,
            'layer': layer,
            'window': selected_window,
            'score': round(score, 6)
        }))

    def _get_alert_level(self, layer: str, score: float) -> int:
        if layer == 'L2':
            if score > 0.8:
                return 12
            elif score > 0.5:
                return 10
            else:
                return 8
        else:
            if score > 0.1:
                return 7
            elif score > 0.05:
                return 5
            else:
                return 3

    def _get_severity(self, score: float) -> str:
        if score > 0.1:
            return 'critical'
        elif score > 0.05:
            return 'high'
        elif score > 0.02:
            return 'medium'
        else:
            return 'low'
