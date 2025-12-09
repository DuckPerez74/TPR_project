import json
import logging
from datetime import datetime
from pathlib import Path
from config import ANOMALY_LOG_PATH

class WazuhAnomalyLogger:
    def __init__(self, log_path=ANOMALY_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger('wazuh_anomaly')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    def log_anomaly(self, entity_id, window_minutes, anomaly_score, metrics_summary=None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "anomaly_detection",
            "severity": self._calculate_severity(anomaly_score),
            "entity_id": entity_id,
            "observation_window_minutes": window_minutes,
            "anomaly_score": float(anomaly_score),
            "description": f"Anomaly detected for entity {entity_id} in {window_minutes}min window"
        }

        if metrics_summary:
            log_entry["metrics_summary"] = metrics_summary

        self.logger.info(json.dumps(log_entry))

    def log_detection_run(self, entities_analyzed, anomalies_found):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "detection_run",
            "entities_analyzed": entities_analyzed,
            "anomalies_detected": anomalies_found
        }

        self.logger.info(json.dumps(log_entry))

    def _calculate_severity(self, score):
        if score > 0.1:
            return "critical"
        elif score > 0.05:
            return "high"
        elif score > 0.02:
            return "medium"
        else:
            return "low"
