# Anomaly Logging - Formato de Alertas TPR

## Visão Geral

O wodle-tpr implementa um sistema de logging hierárquico que:
- Detecta anomalias nas janelas de 60min → 30min → 10min (drill-down)
- **Registra apenas na menor janela que detectou anomalia**
- Diferencia criticidade por layer: **L1 = baixa/média**, **L2 = alta/crítica**
- Gera JSON formatado para ingestão pelo Wazuh

## Ficheiros de Log

```
/var/ossec/logs/anomaly_detection.log  # Log operacional (debug, métricas, erros)
/var/ossec/logs/anomaly.log            # Alertas de anomalia (formato Wazuh)
```

## Lógica de Drill-Down

### Exemplo 1: Anomalia em todas as janelas
```
60min: ANOMALIA (score=0.15)
30min: ANOMALIA (score=0.18)
10min: ANOMALIA (score=0.22)

→ Regista alerta: window=10 (menor janela)
```

### Exemplo 2: Anomalia só em 60min e 30min
```
60min: ANOMALIA (score=0.12)
30min: ANOMALIA (score=0.14)
10min: NORMAL (score=0.008)

→ Regista alerta: window=30 (menor janela com anomalia)
```

### Exemplo 3: Anomalia só em 60min
```
60min: ANOMALIA (score=0.09)
30min: NORMAL (score=0.015)
10min: NORMAL (score=0.012)

→ Regista alerta: window=60 (única janela com anomalia)
```

## Formato JSON - Layer L1 (Entity-Level)

```json
{
  "timestamp": "2024-12-12T14:30:00.000Z",
  "rule": {
    "id": "100001",
    "level": 5,
    "description": "TPR Anomaly Detected - Layer L1",
    "groups": ["tpr", "anomaly_detection", "layer_l1"]
  },
  "data": {
    "tpr": {
      "entity_id": "436",
      "observation_window_minutes": 30,
      "layer": "L1",
      "anomaly_score": 0.082345,
      "model_used": "entity_436",
      "cluster_id": null,
      "drill_down": {
        "window_60": true,
        "window_30": true,
        "window_10": false
      }
    }
  },
  "decoder": {
    "name": "tpr-anomaly"
  },
  "location": "wodle-tpr"
}
```

## Formato JSON - Layer L2 (User/IP/Route)

### Anomalia de User
```json
{
  "timestamp": "2024-12-12T14:30:00.000Z",
  "rule": {
    "id": "100002",
    "level": 10,
    "description": "TPR User Anomaly Detected - john.doe@company.com",
    "groups": ["tpr", "anomaly_detection", "layer_l2"]
  },
  "data": {
    "tpr": {
      "entity_id": "436",
      "observation_window_minutes": 10,
      "layer": "L2",
      "anomaly_score": 0.756821,
      "model_used": "user_john_doe_company_com",
      "cluster_id": null,
      "l2_dimension": "user",
      "l2_dimension_value": "john.doe@company.com",
      "l2_all_anomalies": [
        {
          "dimension": "user",
          "dimension_value": "john.doe@company.com",
          "score": 0.756821,
          "model_used": "user_john_doe_company_com"
        },
        {
          "dimension": "user",
          "dimension_value": "jane.smith@company.com",
          "score": 0.512345,
          "model_used": "l2_simple_fallback"
        }
      ],
      "drill_down": {
        "window_60": true,
        "window_30": true,
        "window_10": true
      }
    }
  },
  "decoder": {
    "name": "tpr-anomaly"
  },
  "location": "wodle-tpr"
}
```

### Anomalia de Source IP
```json
{
  "timestamp": "2024-12-12T14:30:00.000Z",
  "rule": {
    "id": "100002",
    "level": 12,
    "description": "TPR Source IP Anomaly Detected - 192.168.1.100",
    "groups": ["tpr", "anomaly_detection", "layer_l2"]
  },
  "data": {
    "tpr": {
      "entity_id": "436",
      "observation_window_minutes": 30,
      "layer": "L2",
      "anomaly_score": 0.923456,
      "model_used": null,
      "cluster_id": null,
      "l2_dimension": "source_ip",
      "l2_dimension_value": "192.168.1.100",
      "l2_all_anomalies": [
        {
          "dimension": "source_ip",
          "dimension_value": "192.168.1.100",
          "score": 0.923456,
          "model_used": "l2_simple_fallback"
        }
      ],
      "drill_down": {
        "window_60": true,
        "window_30": true,
        "window_10": false
      }
    }
  },
  "decoder": {
    "name": "tpr-anomaly"
  },
  "location": "wodle-tpr"
}
```

### Anomalia de Route
```json
{
  "timestamp": "2024-12-12T14:30:00.000Z",
  "rule": {
    "id": "100002",
    "level": 8,
    "description": "TPR Route Anomaly Detected - /api/v1/admin/users",
    "groups": ["tpr", "anomaly_detection", "layer_l2"]
  },
  "data": {
    "tpr": {
      "entity_id": "436",
      "observation_window_minutes": 60,
      "layer": "L2",
      "anomaly_score": 0.634512,
      "model_used": null,
      "cluster_id": null,
      "l2_dimension": "route",
      "l2_dimension_value": "/api/v1/admin/users",
      "l2_all_anomalies": [
        {
          "dimension": "route",
          "dimension_value": "/api/v1/admin/users",
          "score": 0.634512,
          "model_used": "l2_simple_fallback"
        }
      ],
      "drill_down": {
        "window_60": true,
        "window_30": false,
        "window_10": false
      }
    }
  },
  "decoder": {
    "name": "tpr-anomaly"
  },
  "location": "wodle-tpr"
}
```

## Níveis de Severidade (Rule Level)

### Layer L1 (Entity-Level)
```
score > 0.1  → level 7  (medium-high)
score > 0.05 → level 5  (medium)
score ≤ 0.05 → level 3  (low)
```

### Layer L2 (User/IP/Route)
```
score > 0.8  → level 12 (critical)
score > 0.5  → level 10 (high)
score ≤ 0.5  → level 8  (medium-high)
```

## Campos Principais

| Campo | Descrição |
|-------|-----------|
| `timestamp` | Timestamp UTC do alerta (ISO 8601) |
| `rule.id` | 100001=L1, 100002=L2 |
| `rule.level` | Severidade Wazuh (3-12) |
| `rule.description` | Descrição human-readable |
| `data.tpr.entity_id` | ID da entidade anómala |
| `data.tpr.observation_window_minutes` | Menor janela com anomalia (60/30/10) |
| `data.tpr.layer` | L1 ou L2 |
| `data.tpr.anomaly_score` | Score de anomalia (0-1+) |
| `data.tpr.model_used` | Modelo usado (entity_X, cluster_Y, user_X, l2_simple_fallback) |
| `data.tpr.l2_dimension` | Dimensão L2 (user/source_ip/route) |
| `data.tpr.l2_dimension_value` | Valor específico (email, IP, URL) |
| `data.tpr.l2_all_anomalies` | Lista de TODAS as anomalias L2 detectadas |
| `data.tpr.drill_down` | Resultado de cada janela (60/30/10) |

## Configuração Wazuh

### 1. Configurar ossec.conf para monitorizar anomaly.log

```xml
<ossec_config>
  <localfile>
    <log_format>json</log_format>
    <location>/var/ossec/logs/anomaly.log</location>
  </localfile>
</ossec_config>
```

### 2. Criar Regras no Wazuh (local_rules.xml)

```xml
<group name="tpr,anomaly_detection">

  <!-- L1 Anomalies -->
  <rule id="100001" level="0">
    <decoded_as>json</decoded_as>
    <field name="rule.id">100001</field>
    <description>TPR Layer 1 (Entity-Level) Anomaly Detected</description>
    <group>tpr,layer_l1</group>
  </rule>

  <rule id="100011" level="3">
    <if_sid>100001</if_sid>
    <field name="data.tpr.anomaly_score">\.0\d</field>
    <description>TPR L1 Low Severity Anomaly - Entity $(data.tpr.entity_id)</description>
    <group>tpr,layer_l1,low</group>
  </rule>

  <rule id="100012" level="5">
    <if_sid>100001</if_sid>
    <field name="data.tpr.anomaly_score">\.0[5-9]</field>
    <description>TPR L1 Medium Severity Anomaly - Entity $(data.tpr.entity_id)</description>
    <group>tpr,layer_l1,medium</group>
  </rule>

  <rule id="100013" level="7">
    <if_sid>100001</if_sid>
    <field name="data.tpr.anomaly_score">\.[1-9]</field>
    <description>TPR L1 High Severity Anomaly - Entity $(data.tpr.entity_id)</description>
    <group>tpr,layer_l1,high</group>
  </rule>

  <!-- L2 Anomalies -->
  <rule id="100002" level="0">
    <decoded_as>json</decoded_as>
    <field name="rule.id">100002</field>
    <description>TPR Layer 2 (User/IP/Route) Anomaly Detected</description>
    <group>tpr,layer_l2</group>
  </rule>

  <!-- L2 User Anomalies -->
  <rule id="100021" level="8">
    <if_sid>100002</if_sid>
    <field name="data.tpr.l2_dimension">user</field>
    <field name="data.tpr.anomaly_score">\.5</field>
    <description>TPR L2 User Anomaly (Medium-High) - $(data.tpr.l2_dimension_value)</description>
    <group>tpr,layer_l2,user,medium-high</group>
  </rule>

  <rule id="100022" level="10">
    <if_sid>100002</if_sid>
    <field name="data.tpr.l2_dimension">user</field>
    <field name="data.tpr.anomaly_score">\.[5-7]</field>
    <description>TPR L2 User Anomaly (High) - $(data.tpr.l2_dimension_value)</description>
    <group>tpr,layer_l2,user,high</group>
  </rule>

  <rule id="100023" level="12">
    <if_sid>100002</if_sid>
    <field name="data.tpr.l2_dimension">user</field>
    <field name="data.tpr.anomaly_score">\.[8-9]|^[1-9]</field>
    <description>TPR L2 User Anomaly (Critical) - $(data.tpr.l2_dimension_value)</description>
    <group>tpr,layer_l2,user,critical</group>
  </rule>

  <!-- L2 Source IP Anomalies -->
  <rule id="100031" level="10">
    <if_sid>100002</if_sid>
    <field name="data.tpr.l2_dimension">source_ip</field>
    <description>TPR L2 Source IP Anomaly - $(data.tpr.l2_dimension_value)</description>
    <group>tpr,layer_l2,source_ip</group>
  </rule>

  <!-- L2 Route Anomalies -->
  <rule id="100041" level="8">
    <if_sid>100002</if_sid>
    <field name="data.tpr.l2_dimension">route</field>
    <description>TPR L2 Route Anomaly - $(data.tpr.l2_dimension_value)</description>
    <group>tpr,layer_l2,route</group>
  </rule>

</group>
```

### 3. Decoders (local_decoder.xml)

```xml
<decoder name="tpr-anomaly">
  <program_name>^wodle-tpr$</program_name>
</decoder>

<decoder name="tpr-anomaly-json">
  <parent>tpr-anomaly</parent>
  <type>json</type>
</decoder>
```

## Monitorização e Troubleshooting

### Verificar alertas gerados
```bash
tail -f /var/ossec/logs/anomaly.log | jq .
```

### Contar alertas por layer
```bash
grep -c '"layer": "L1"' /var/ossec/logs/anomaly.log
grep -c '"layer": "L2"' /var/ossec/logs/anomaly.log
```

### Ver últimas anomalias de users
```bash
cat /var/ossec/logs/anomaly.log | jq 'select(.data.tpr.l2_dimension == "user") | {timestamp, user: .data.tpr.l2_dimension_value, score: .data.tpr.anomaly_score, window: .data.tpr.observation_window_minutes}'
```

### Verificar drill-down
```bash
cat /var/ossec/logs/anomaly.log | jq '{entity: .data.tpr.entity_id, window: .data.tpr.observation_window_minutes, drill_down: .data.tpr.drill_down}'
```

## Integração com SIEM

O formato JSON é compatível com:
- **Wazuh SIEM** (nativo)
- **Elastic Stack** (via Filebeat)
- **Splunk** (via HEC ou file monitoring)
- **QRadar** (via syslog JSON)

## Performance

- **1 alerta por anomalia** detectada (não duplica em múltiplas janelas)
- Rotation automático (100MB, 5 backups)
- Formato JSON compacto (~500-800 bytes por alerta)
