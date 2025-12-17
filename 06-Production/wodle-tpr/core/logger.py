import json
import logging
import socket
import uuid
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


# Message header for Wazuh queue (same pattern as AWS wodle)
MESSAGE_HEADER = "1:Wazuh-TPR:"
WAZUH_QUEUE_PATH = "/var/ossec/queue/sockets/queue"
MAX_EVENT_SIZE = 65535  # Maximum event size for Wazuh


class WazuhLogger:
    """
    Logger for TPR Anomaly Detection Wodle.

    Sends events to Wazuh via socket (like AWS wodle) and maintains
    local file logging for debugging/troubleshooting.
    """

    def __init__(self, config: dict):
        log_config = config.get('logging', {})

        # File paths for local logging (debugging)
        self.log_file = Path(log_config.get('file', '/var/ossec/logs/anomaly_detection.log'))
        self.anomaly_log_file = Path(log_config.get('anomaly_file', '/var/ossec/logs/anomaly.log'))
        self.level = getattr(logging, log_config.get('level', 'INFO'))

        # Option to disable file logging entirely
        self.enable_file_logging = log_config.get('enable_file_logging', True)

        # Wazuh queue path (can be overridden in config)
        self.wazuh_queue = log_config.get('wazuh_queue', WAZUH_QUEUE_PATH)

        # Setup internal logger for operational logs
        self.logger = logging.getLogger('anomaly_detection')
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()

        if self.enable_file_logging:
            max_bytes = log_config.get('max_size_mb', 100) * 1024 * 1024
            backup_count = log_config.get('backup_count', 5)

            # Create parent directories if they don't exist
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.anomaly_log_file.parent.mkdir(parents=True, exist_ok=True)

            handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            handler.setLevel(self.level)
            self.logger.addHandler(handler)
        else:
            # Use NullHandler when file logging is disabled
            self.logger.addHandler(logging.NullHandler())

        # Anomaly logger for local file backup (fallback)
        self.anomaly_logger = logging.getLogger('anomaly_alerts')
        self.anomaly_logger.setLevel(logging.INFO)
        self.anomaly_logger.handlers.clear()
        self.anomaly_logger.propagate = False

        if self.enable_file_logging:
            anomaly_handler = RotatingFileHandler(
                self.anomaly_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            anomaly_handler.setLevel(logging.INFO)
            anomaly_handler.setFormatter(logging.Formatter('%(message)s'))
            self.anomaly_logger.addHandler(anomaly_handler)
        else:
            self.anomaly_logger.addHandler(logging.NullHandler())

    def send_to_wazuh(self, msg: dict) -> bool:
        """
        Send event directly to Wazuh queue via Unix socket.

        This is the same mechanism used by the AWS wodle.
        Falls back to file logging if socket is unavailable (e.g., Windows testing).

        Args:
            msg: Dictionary with event data

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            json_msg = json.dumps(msg, default=str)
            encoded_msg = f"{MESSAGE_HEADER}{json_msg}".encode()

            # Check message size
            if len(encoded_msg) > MAX_EVENT_SIZE:
                self.logger.warning(
                    f"Event size ({len(encoded_msg)} bytes) exceeds maximum ({MAX_EVENT_SIZE} bytes)"
                )

            # Try Unix socket (Linux/Wazuh environment)
            s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            s.connect(self.wazuh_queue)
            s.send(encoded_msg)
            s.close()
            return True

        except (socket.error, OSError) as e:
            # Socket not available (Windows or Wazuh not running)
            if hasattr(e, 'errno'):
                if e.errno == 111:
                    self.logger.debug("Wazuh not running, falling back to file logging")
                elif e.errno == 90:
                    self.logger.error("Message too long for Wazuh socket buffer")
                else:
                    self.logger.debug(f"Socket error ({e.errno}), falling back to file logging")
            else:
                self.logger.debug(f"Socket unavailable: {e}, falling back to file logging")

            # Fallback: write to local anomaly log file
            self.anomaly_logger.info(json.dumps(msg))
            return False

        except Exception as e:
            self.logger.error(f"Error sending to Wazuh: {e}")
            self.anomaly_logger.info(json.dumps(msg))
            return False

    def log_anomaly_alert(self, entity_id: str, analysis_result: dict) -> str:
        """
        Log anomaly alert to Wazuh (via socket) and return alert_id for correlation.

        Message format follows AWS wodle pattern - no rule/level info.
        Alert levels are determined by Wazuh rules (0450-tpr_rules.xml).
        """
        selected_window = analysis_result.get('selected_window')
        window_result = analysis_result['results'][str(selected_window)]

        layer = window_result['anomaly_layer']
        score = window_result['score']
        alert_id = str(uuid.uuid4())

        # Build TPR event (AWS wodle style - no rule/level)
        event = {
            'integration': 'tpr',
            'tpr': {
                'event_type': 'anomaly',
                'alert_id': alert_id,
                'entity_id': str(entity_id),
                'layer': layer,
                'anomaly_score': round(score, 6),
                'observation_window_minutes': selected_window,
                'model_used': window_result.get('model_used'),
                'cluster_id': window_result.get('cluster_id')
            }
        }

        # Add risk scoring data if available (as percentage 0-100%)
        if 'risk_score' in analysis_result:
            event['tpr']['risk_score'] = round(analysis_result['risk_score'] * 100)

            # Add risk components breakdown (as percentages, rounded)
            if 'risk_components' in analysis_result:
                components = analysis_result['risk_components']
                event['tpr']['risk_l1'] = round(components.get('l1_weighted', 0.0) * 100)
                event['tpr']['risk_l2_user'] = round(components.get('l2_user_weighted', 0.0) * 100)
                event['tpr']['risk_l2_route'] = round(components.get('l2_route_weighted', 0.0) * 100)

        # Add L2-specific fields
        if layer == 'L2':
            event['tpr']['l2_dimension'] = window_result.get('l2_dimension')
            event['tpr']['l2_dimension_value'] = str(window_result.get('l2_dimension_value', ''))

            # Convert l2_details list to summary string (avoid nested objects)
            l2_details = window_result.get('l2_details', [])
            if l2_details:
                event['tpr']['l2_anomalies_count'] = len(l2_details)
                # Summarize top anomalies as pipe-separated values
                event['tpr']['l2_anomalies_summary'] = '|'.join(
                    [f"{d.get('dimension', 'unknown')}:{d.get('score', 0):.4f}" for d in l2_details[:5]]
                )

        # Send to Wazuh via socket
        self.send_to_wazuh(event)

        # Also log to internal operational log
        self.logger.info(json.dumps({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'anomaly_alert_sent',
            'alert_id': alert_id,
            'entity_id': entity_id,
            'layer': layer,
            'window': selected_window,
            'score': round(score, 6)
        }))

        return alert_id

    def log_llm_analysis(self, entity_id: str, llm_result: dict, alert_id: str = None):
        """
        Log LLM analysis result to Wazuh (via socket).

        Message format follows AWS wodle pattern - no rule/level info.
        Alert levels are determined by Wazuh rules based on classification.
        """
        # Build TPR LLM event (AWS wodle style - no rule/level)
        event = {
            'integration': 'tpr',
            'tpr': {
                'event_type': 'llm_analysis',
                'alert_id': alert_id or str(uuid.uuid4()),
                'entity_id': str(entity_id),
                'classification': llm_result.get('classification', 'Unknown'),
                'threat_type': llm_result.get('threat_type', 'Unknown'),
                'confidence': llm_result.get('confidence', 'Unknown'),
                'explanation': llm_result.get('explanation', ''),
                'model': llm_result.get('model', ''),
                'logs_analyzed': llm_result.get('logs_analyzed', 0),
                'execution_time_ms': llm_result.get('execution_time_ms', 0)
            }
        }

        # Convert arrays to pipe-separated strings (avoid Elasticsearch mapping conflicts)
        user_operations = llm_result.get('user_operations', [])
        if user_operations:
            event['tpr']['user_operations'] = '|'.join(str(op) for op in user_operations[:10])

        recommended_actions = llm_result.get('recommended_actions', [])
        if recommended_actions:
            event['tpr']['recommended_actions'] = '|'.join(str(action) for action in recommended_actions[:5])

        # Send to Wazuh via socket
        self.send_to_wazuh(event)

        # Also log to internal operational log
        self.logger.info(json.dumps({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'llm_analysis_sent',
            'entity_id': entity_id,
            'alert_id': alert_id,
            'classification': llm_result.get('classification'),
            'threat_type': llm_result.get('threat_type')
        }))

    def log_anomaly(self, entity_id: str, window: int, score: float,
                    layer: str, metrics_summary: dict):
        """Log internal anomaly detection metrics (for debugging)."""
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

        self.logger.info(json.dumps(log_entry))

    def log_detection_run(self, entities_analyzed: int, anomalies_found: int,
                          execution_time_ms: int):
        """Log detection run statistics."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'detection_run',
            'entities_analyzed': entities_analyzed,
            'anomalies_found': anomalies_found,
            'execution_time_ms': execution_time_ms
        }

        self.logger.info(json.dumps(log_entry))

    def log_error(self, message: str, error: Exception = None):
        """Log error messages."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'error',
            'message': message
        }

        if error:
            log_entry['error'] = str(error)
            log_entry['error_type'] = type(error).__name__

        self.logger.error(json.dumps(log_entry))

    def _get_severity(self, score: float) -> str:
        """Map anomaly score to severity label."""
        if score > 0.1:
            return 'critical'
        elif score > 0.05:
            return 'high'
        elif score > 0.02:
            return 'medium'
        else:
            return 'low'
