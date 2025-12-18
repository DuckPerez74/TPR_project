import json
import logging
import socket
import uuid
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

MESSAGE_HEADER = "1:Wazuh-TPR:"
WAZUH_QUEUE_PATH = "/var/ossec/queue/sockets/queue"
MAX_EVENT_SIZE = 65535


class WazuhLogger:

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
        json_msg = json.dumps(msg, default=str)
        socket_success = False
        
        try:
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
            socket_success = True

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

        except Exception as e:
            self.logger.error(f"Error sending to Wazuh: {e}")

        # Always save to anomaly.log when enable_file_logging is true
        # This provides a local backup of all alerts sent to Wazuh
        if self.enable_file_logging:
            self.anomaly_logger.info(json_msg)
        
        return socket_success

    def log_anomaly_alert(self, entity_id: str, analysis_result: dict) -> str:
        selected_window = analysis_result.get('selected_window')
        results = analysis_result.get('results', {})
        window_result = results[str(selected_window)]
        alert_id = str(uuid.uuid4())
        
        # Calculate risk score (as percentage)
        risk_score = round(analysis_result.get('risk_score', 0.0) * 100)
        
        # Determine severity based on risk_score
        if risk_score >= 60:
            severity = 'critical'
        elif risk_score >= 40:
            severity = 'high'
        elif risk_score >= 20:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Find which windows had anomalies
        windows_affected = []
        for win in ['10', '30', '60']:
            if results.get(win, {}).get('has_anomaly', False):
                windows_affected.append(int(win))
        
        # Count L2 users and routes from selected window
        l2_details = window_result.get('l2_details', [])
        l2_users_count = sum(1 for d in l2_details if d.get('dimension') == 'user')
        l2_routes_count = sum(1 for d in l2_details if d.get('dimension') == 'route')
        
        # Build triggers summary
        triggers = {
            'l1': window_result.get('l1_anomaly', False),
            'l2_users': l2_users_count,
            'l2_routes': l2_routes_count
        }
        
        # Build top impacts (max 5, sorted by score)
        top_impacts = []
        if l2_details:
            sorted_details = sorted(l2_details, key=lambda x: x.get('score', 0), reverse=True)
            for detail in sorted_details[:5]:
                top_impacts.append({
                    'type': detail.get('dimension', 'unknown'),
                    'value': str(detail.get('dimension_value', '')),
                    'score': round(detail.get('score', 0) * 100)  # as percentage
                })
        
        # Build simplified TPR event
        event = {
            'integration': 'tpr',
            'tpr': {
                'event_type': 'anomaly',
                'alert_id': alert_id,
                'entity_id': str(entity_id),
                
                # Quick triaging
                'risk_score': risk_score,
                'severity': severity,
                'windows_affected': windows_affected,
                'selected_window': selected_window,
                
                # What happened
                'triggers': triggers,
                
                # Top impacted (for quick investigation)
                'top_impacts': top_impacts if top_impacts else None
            }
        }
        
        # Add model info if available
        model_used = window_result.get('model_used')
        if model_used:
            event['tpr']['model_used'] = model_used

        # Send to Wazuh via socket
        self.send_to_wazuh(event)

        # Also log to internal operational log
        self.logger.info(json.dumps({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'anomaly_alert_sent',
            'alert_id': alert_id,
            'entity_id': entity_id,
            'risk_score': risk_score,
            'severity': severity,
            'windows_affected': windows_affected
        }))

        return alert_id

    def log_llm_analysis(self, entity_id: str, llm_result: dict, alert_id: str = None):
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

        # Include arrays as structured JSON (Wazuh JSON decoder handles nested objects)
        user_operations = llm_result.get('user_operations', [])
        if user_operations:
            event['tpr']['user_operations'] = user_operations[:10]

        recommended_actions = llm_result.get('recommended_actions', [])
        if recommended_actions:
            event['tpr']['recommended_actions'] = recommended_actions[:5]

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

    def update_alert_with_llm(self, alert_id: str, llm_result: dict) -> bool:
        entity_id = llm_result.get('entity_id', 'unknown')
        self.logger.info(f"Updating alert {alert_id} with LLM result for entity {entity_id}")
        
        try:
            from core.opensearch_client import OpenSearchClient
            client = OpenSearchClient.get_instance()

            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"data.tpr.alert_id": alert_id}},
                            {"term": {"data.tpr.event_type": "anomaly"}}
                        ]
                    }
                },
                "size": 1,
                "sort": [{"@timestamp": "desc"}]
            }

            # Search in recent wazuh-alerts indices
            today = datetime.utcnow().strftime("%Y.%m.%d")
            index_pattern = f"wazuh-alerts-4.x-{today}"
            
            self.logger.info(f"Searching for alert_id {alert_id} in index {index_pattern}")

            response = client.search(index=index_pattern, body=search_body)
            hits = response.get('hits', {}).get('hits', [])
            
            self.logger.info(f"Search returned {len(hits)} hits for alert_id {alert_id}")

            if not hits:
                self.logger.warning(f"Could not find alert document for alert_id: {alert_id}, sending via fallback")
                # Fallback: send as separate log via Wazuh socket
                self.log_llm_analysis(entity_id, llm_result, alert_id)
                return False

            doc_id = hits[0]['_id']
            doc_index = hits[0]['_index']
            
            self.logger.info(f"Found document {doc_id} in {doc_index} for alert_id {alert_id}")

            # Build LLM analysis update
            llm_analysis = {
                'classification': llm_result.get('classification', 'Unknown'),
                'threat_type': llm_result.get('threat_type', 'Unknown'),
                'confidence': llm_result.get('confidence', 'Unknown'),
                'explanation': llm_result.get('explanation', ''),
                'model': llm_result.get('model', ''),
                'logs_analyzed': llm_result.get('logs_analyzed', 0),
                'execution_time_ms': llm_result.get('execution_time_ms', 0)
            }

            # Include user operations (max 10)
            user_operations = llm_result.get('user_operations', [])
            if user_operations:
                llm_analysis['user_operations'] = user_operations[:10]

            # Include recommended actions (max 5)
            recommended_actions = llm_result.get('recommended_actions', [])
            if recommended_actions:
                llm_analysis['recommended_actions'] = recommended_actions[:5]

            # Update the document
            update_body = {
                "doc": {
                    "data": {
                        "tpr": {
                            "llm_analysis": llm_analysis
                        }
                    }
                }
            }

            client.update(index=doc_index, id=doc_id, body=update_body)
            
            self.logger.info(f"Successfully updated OpenSearch document {doc_id} with LLM analysis for entity {entity_id}")

            self.logger.info(json.dumps({
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'event_type': 'alert_updated_with_llm',
                'alert_id': alert_id,
                'doc_id': doc_id,
                'classification': llm_result.get('classification')
            }))

            return True

        except Exception as e:
            self.logger.error(f"Failed to update alert with LLM analysis for entity {entity_id}: {e}")
            # Fallback: send as separate log
            self.log_llm_analysis(entity_id, llm_result, alert_id)
            return False

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
