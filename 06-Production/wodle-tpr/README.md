# Wodle TPR - Anomaly Detection for Wazuh

Standalone multi-layered anomaly detection system for Wazuh SIEM.

## Features

- L1 (entity-level) + L2 (user/IP/route-level) metrics calculation
- PyTorch-based autoencoder models
- K-means clustering for entity profiling
- Entity-specific and cluster-based anomaly detection
- Hierarchical drill-down (60m → 30m → 10m)
- Unified metrics index (metrics-tpr)
- Automatic warmup/production mode switching

## Architecture

```
wodle-tpr/
├── main.py              # Real-time detection (runs every 1 min)
├── warmup.py            # Historical data processing
├── train.py             # Model training orchestrator
├── core/                # Configuration, scheduling, logging
├── data/                # Log fetching & preprocessing
├── metrics/             # L1 + L2 metrics (45 L1 features)
├── detection/           # PyTorch anomaly detection
├── training/            # AutoEncoder + K-means training
└── models/              # Trained models (.pt files)
```

## Deployment Workflow

### Phase 1: Installation

```bash
chmod +x install.sh
sudo ./install.sh
```

### Phase 2: Warmup (Historical Data)

Configure warmup period in `config.json`:
```json
"warmup": {
  "enabled": true,
  "start_date": "2024-08-01T00:00:00Z",
  "end_date": "2024-11-30T23:59:59Z",
  "batch_hours": 24
}
```

Run warmup:
```bash
sudo python3 /var/ossec/wodles/wodle-tpr/warmup.py
```

This creates `metrics-tpr` index with L1 + L2 metrics.

### Phase 3: Model Training

```bash
sudo python3 /var/ossec/wodles/wodle-tpr/train.py
```

This will:
1. Read metrics from `metrics-tpr`
2. Perform K-means clustering (3 clusters)
3. Train entity models (entities with >100 samples)
4. Train cluster models (fallback for new entities)
5. Calculate thresholds (p90, p95, p99, 3sigma)
6. Save models to `models/`

### Phase 4: Production

Enable wodle in `/var/ossec/etc/ossec.conf`:

```xml
<wodle name="command">
  <disabled>no</disabled>
  <tag>wodle-tpr</tag>
  <command>/var/ossec/wodles/wodle-tpr/main.py</command>
  <interval>1m</interval>
  <run_on_start>yes</run_on_start>
  <timeout>300</timeout>
</wodle>
```

Restart Wazuh:
```bash
systemctl restart wazuh-manager
```

The wodle automatically detects mode:
- **No models** → Warmup mode (metrics only)
- **Models exist** → Production mode (metrics + detection)

## Configuration

### Environment (.env)

```bash
OPENSEARCH_HOST=https://your-host:9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=YourPassword
```

User needs write permissions for `metrics-tpr` index.

### Advanced Settings (config.json)

- Observation windows (10m, 30m, 60m)
- L1/L2 layer configuration
- Detection thresholds
- Model paths
- Logging settings

## L1 Metrics (46 features)

Based on POC specifications:

**Request Patterns (7)**
- total_requests, mean/max/std_requests_per_minute
- cv_request_rate, peak_to_average_ratio, burst_score

**Response Patterns (8)**
- pct_2xx/3xx/4xx/5xx_responses, error_rate, critical_error_rate
- status_code_entropy, unique_status_codes

**Timing Patterns (9)**
- mean/std_response_time, p50/p75/p90/p95/p99_response_time
- pct_slow/very_slow_requests

**Source Patterns (5)**
- unique_source_ips, mean/max_requests_per_ip
- gini_ip_distribution, ip_concentration_top10pct

**Route Patterns (7)**
- unique_api_modules/routes, module/route_entropy
- top_module/top5_routes_percentage, module_switching_frequency

**Payload Patterns (4)**
- mean/std/max/min_response_size

**User Agent Patterns (3)**
- unique_user_agents, user_agent_entropy, bot_like_ua_percentage

**HTTP Patterns (2)**
- unique_http_methods, get/post_request_ratio

## Field Mapping

POC field names (Wazuh logs):
- `data.status_code` - HTTP status
- `data.response_time` - Response time (seconds)
- `data.srcip` - Source IP
- `data.api_module` - API module
- `data.route_uri` - Route URI
- `data.size` - Response size (bytes)
- `data.browser` - User agent
- `data.method` - HTTP method

## Monitoring

```bash
tail -f /var/ossec/logs/anomaly_detection.log

curl -X GET "https://opensearch:9200/metrics-tpr/_count" -u admin:password
```

## Model Files

After training:
```
models/
├── entity_models/
│   ├── entity_436.pt
│   ├── entity_436_scaler.pkl
│   ├── entity_436_thresholds.json
│   └── ...
├── cluster_models/
│   ├── cluster_0.pt
│   ├── cluster_0_scaler.pkl
│   ├── cluster_0_thresholds.json
│   └── ...
└── kmeans_clusterer.pkl
```

## Technology Stack

- **PyTorch** - AutoEncoder models
- **scikit-learn** - K-means clustering, preprocessing
- **pandas** - Data manipulation
- **OpenSearch** - Metrics storage & log source

## License

Internal use only.




Opção A (20 min)

(5min) Alex => Métricas (Features) 
(5min) Diogo => Cluster (K-means) + L1 (AutoEncoder) + L2 (Isolation Forest - User/Route)
(4min) Daniel => Pipeline (Wodle) 

(5min) DEMO


- Preparar a apresentação
- Treinar os modelos
- Colocar em produção
- Simular ataques (DEMO)


LLM 
