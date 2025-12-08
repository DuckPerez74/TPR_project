# Auto-Encoder L1 Training

Train cluster-specific or entity-specific auto-encoders for L1 anomaly detection.

## Quick Start

### Single Command Training

Use `train.py` for all training needs:

```bash
# Train ONE entity model
python train.py single --entity-id 123 --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-09-30T00:00:00

# Train ONE cluster model
python train.py single --cluster-id 0 \
    --entity-file ../../03-Clusters/cluster0_entities.csv \
    --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-09-30T00:00:00

# Train ALL entities (from OpenSearch)
python train.py all-entities --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-09-30T00:00:00 \
    --skip-existing

# Train ALL clusters
python train.py all-clusters --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-09-30T00:00:00 \
    --skip-existing
```

## Output Structure

Models are saved in organized directories:

```
out/
├── entity-10/
│   ├── entity-10-60m.pth
│   ├── entity-10-60m_scaler.pkl
│   ├── entity-10-60m_features.json
│   ├── entity-10-60m_thresholds.json
│   └── ...
│
├── cluster-0/
│   ├── cluster-0-60m.pth
│   ├── cluster-0-60m_scaler.pkl
│   └── ...
│
└── training_report_*.txt
```

## Common Use Cases

### Test with Limited Entities

```bash
# Train first 5 entities only
python train.py all-entities --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-08-02T00:00:00 \
    --limit 5
```

### Resume Training

```bash
# Skip already trained models
python train.py all-entities --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-09-30T00:00:00 \
    --skip-existing

# Resume from entity index 100
python train.py all-entities --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-09-30T00:00:00 \
    --start-from 100 --skip-existing
```

### Production Training

```bash
# Train all entities with test set
python train.py all-entities --window 60 \
    --train-start 2025-08-01T00:00:00 --train-end 2025-09-30T00:00:00 \
    --test-start 2025-10-01T00:00:00 --test-end 2025-11-30T00:00:00 \
    --skip-existing --device cuda
```

## Parameters

### Required
- `--window`: Time window (10, 30, or 60 minutes)
- `--train-start`: Training start date (ISO 8601 format)
- `--train-end`: Training end date (ISO 8601 format)

### Optional
- `--test-start`: Test start date (ISO 8601)
- `--test-end`: Test end date (ISO 8601)
- `--output-dir`: Output directory (default: `out`)
- `--device`: Device to use (`auto`, `cuda`, `cpu` - default: `auto`)
- `--skip-existing`: Skip already trained models
- `--limit`: Limit number of entities (for testing)
- `--start-from`: Resume from entity index

### Mode-Specific

**single mode**:
- `--entity-id`: Entity/company ID (for entity model)
- `--cluster-id`: Cluster ID (for cluster model)
- `--entity-file`: CSV file with company_ids (required for cluster mode)

**all-clusters mode**:
- `--clusters-dir`: Directory with cluster CSV files (default: `../../03-Clusters`)

## Files

- **`train.py`** - Unified training script (use this!)
- **`train_autoencoder.py`** - Core training logic (called by train.py)
- **`train_all_from_opensearch.py`** - Fetch entities from OpenSearch (called by train.py)
- **`train_cluster_models.py`** - Batch cluster training (called by train.py)
- **`analyze_entity_data.py`** - Analyze entity data availability
- **`train_autoencoder_l1.ipynb`** - Jupyter notebook for interactive training

## Interactive Training

For exploration and visualization, use the Jupyter notebook:

```bash
jupyter notebook train_autoencoder_l1.ipynb
```

## Entity Data Analysis

Check which entities have sufficient data before training:

```bash
python analyze_entity_data.py --window 60 \
    --start 2025-08-01T00:00:00 --end 2025-09-30T00:00:00 \
    --min-samples 2000
```

Output: `entity_data_analysis.csv` with recommendations for entity-specific vs cluster models.

## Model Artifacts

Each trained model produces:

1. `{name}.pth` - PyTorch model
2. `{name}_scaler.pkl` - MinMaxScaler for normalization
3. `{name}_features.json` - Feature list (46 L1 features)
4. `{name}_thresholds.json` - Anomaly thresholds (p90, p95, p99, 3-sigma)
5. `{name}_metadata.json` - Training metadata
6. `{name}_training_history.csv` - Loss curves
7. `{name}_training_history.png` - Convergence plot

## Thresholds

**Entity-specific thresholds**: Each model has calibrated thresholds in `{name}_thresholds.json`:

```json
{
  "p90": 0.002156,
  "p95": 0.002762,
  "p99": 0.004521,
  "mean_plus_3std": 0.007864
}
```

**Use entity-specific thresholds in production** for best accuracy (fewer false positives/negatives).

## Help

```bash
# Show all options
python train.py --help

# Show mode-specific options
python train.py single --help
python train.py all-entities --help
python train.py all-clusters --help
```

## Troubleshooting

### "No documents found"
- Entity has no data in the specified date range
- Check if entity exists: `python analyze_entity_data.py`

### GPU out of memory
- Use `--device cpu`
- Or reduce batch size in `train_autoencoder.py`

### Training too slow
- Use `--device cuda` for GPU acceleration
- Or train in batches with `--limit` and `--start-from`
