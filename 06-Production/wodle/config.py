import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST')
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD')

if not all([OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASSWORD]):
    raise ValueError("Missing required environment variables in .env file")

INDEX_PATTERN = "wazuh-alerts-4.x-*"
COMPANY_ID_FIELD = "data.entities"

METRICS_INDEX_PREFIX = "metrics-l1"
ANOMALY_LOG_PATH = "/var/ossec/logs/anomaly_detection.log"

OBSERVATION_WINDOWS = [60, 30, 10]
HISTORICAL_WINDOW = 10

MODEL_PATH = Path(__file__).parent.parent.parent / "04-Models" / "L1"
ENTITY_MODEL_PATH = MODEL_PATH / "entity_models"
CLUSTER_MODEL_PATH = MODEL_PATH / "cluster_models"
AUTOENCODER_PATH = MODEL_PATH / "models" / "autoencoder_model.keras"

THRESHOLD_TYPE = "p95"

EXECUTION_SCHEDULE = {
    10: [0, 10, 20, 30, 40, 50],
    30: [0, 30],
    60: [0]
}
