# Wazuh Anomaly Detection Wodle

Production-ready wodle for real-time anomaly detection in Wazuh using L1 metrics.

## Architecture

The wodle is executed by Wazuh Manager every minute but only processes data at specific intervals:
- 10-minute windows: :00, :10, :20, :30, :40, :50
- 30-minute windows: :00, :30
- 60-minute windows: :00

### Model Hierarchy (Fallback System)

Detection follows a priority model selection:
1. **Entity Model** (if exists) - Specific model trained for the entity
2. **Cluster Model** (fallback) - Generic model for the entity's cluster
   - Autoencoder predicts cluster assignment
   - Uses corresponding cluster model

This allows:
- High accuracy for entities with specific models
- Coverage for all entities via cluster models
- Automatic fallback without manual configuration

## Components

- `main.py`: Entry point and orchestration
- `scheduler.py`: Timing control logic
- `data_fetcher.py`: OpenSearch data retrieval
- `metrics_calculator.py`: L1 metrics calculation
- `anomaly_detector.py`: ML-based anomaly detection
- `wazuh_logger.py`: Alert logging for Wazuh
- `config.py`: Configuration management
- `utils.py`: Utility functions

## Installation

### 1. Install the wodle

```bash
chmod +x install.sh
sudo ./install.sh
```

### 2. Configure environment

Edit `/var/ossec/wodles/anomaly-detection/.env`:
```
OPENSEARCH_HOST=https://your-host:9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=YourPassword
```

### 3. Configure Wazuh Manager

Add to `/var/ossec/etc/ossec.conf`:

```xml
<wodle name="command">
  <disabled>no</disabled>
  <tag>anomaly-detection</tag>
  <command>/var/ossec/wodles/anomaly-detection/main.py</command>
  <interval>1m</interval>
  <run_on_start>yes</run_on_start>
  <timeout>300</timeout>
</wodle>
```

### 4. Restart Wazuh Manager

```bash
systemctl restart wazuh-manager
```

### 5. Verify execution

```bash
tail -f /var/ossec/logs/anomaly_detection.log
```

## Detection Logic

1. Fetch active entities from last 10 minutes
2. Calculate and store 10-minute historical metrics (always saved)
3. Calculate 60-minute observation window metrics
   - Saved to index at :00 minutes (for continuous training)
4. Calculate 30-minute observation window metrics
   - Saved to index at :00 and :30 minutes (for continuous training)
5. If anomaly detected in 60-min window:
   - Evaluate 30-minute window
   - Evaluate 10-minute window
6. Select smallest window with detected anomaly (priority: 10 > 30 > 60)
7. Log anomaly to `/var/ossec/logs/anomaly_detection.log`

## Metrics Storage

Metrics are stored in OpenSearch indices for continuous model training:

- `metrics-l1-10m`: 10-minute metrics (saved every 10 minutes)
- `metrics-l1-30m`: 30-minute metrics (saved at :00 and :30)
- `metrics-l1-60m`: 60-minute metrics (saved at :00)

## Log Format

```json
{
  "timestamp": "2025-12-09T10:30:00Z",
  "event_type": "anomaly_detection",
  "severity": "high",
  "entity_id": "436",
  "observation_window_minutes": 30,
  "anomaly_score": 0.0234,
  "description": "Anomaly detected for entity 436 in 30min window",
  "metrics_summary": {
    "total_requests": 1234,
    "error_rate": 5.2,
    "mean_response_time": 0.45,
    "model_used": "entity_436",
    "cluster_id": null
  }
}
```

## Automatic Thresholds

Thresholds are calculated automatically during training for both entity and cluster models based on validation set reconstruction errors:

**Entity models**: `entity_{id}_thresholds.json`
**Cluster models**: `cluster_{id}_thresholds.json`

Each threshold file contains:

```json
{
  "p90": 0.0123,
  "p95": 0.0156,
  "p99": 0.0234,
  "3sigma": 0.0189,
  "mean": 0.0089,
  "std": 0.0033
}
```

The wodle uses `p95` by default (configurable in `config.py` via `THRESHOLD_TYPE`). Anomaly detection compares reconstruction MSE against the threshold:
- `MSE > threshold` = Anomaly
- Higher threshold = less sensitive
- Thresholds update automatically when models are retrained

## Directory Structure

```
04-Models/L1/
├── entity_models/              # Entity-specific models (priority)
│   ├── entity_436.pkl
│   ├── entity_436_thresholds.json
│   └── ...
├── cluster_models/             # Cluster models (fallback)
│   ├── cluster_0.pkl
│   ├── cluster_0_thresholds.json
│   └── ...
└── models/
    └── autoencoder_model.keras # For cluster prediction
```

## Requirements

- Python 3.8+
- Wazuh 4.x
- OpenSearch/Elasticsearch
- Trained L1 models (entity and/or cluster) in `04-Models/L1/`
