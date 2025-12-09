import pandas as pd
import numpy as np
from utils import calculate_entropy, calculate_gini, clean_metrics

def calculate_l1_metrics(df_entity, time_window_minutes):
    if df_entity.empty:
        return {}

    metrics = {}
    total_requests = len(df_entity)
    metrics['total_requests'] = total_requests

    if time_window_minutes > 0:
        metrics['mean_requests_per_minute'] = total_requests / time_window_minutes

        if not pd.api.types.is_datetime64_any_dtype(df_entity['@timestamp']):
            df_entity['@timestamp'] = pd.to_datetime(df_entity['@timestamp'])

        requests_per_minute = df_entity.set_index('@timestamp').resample('1min').size()

        metrics['max_requests_per_minute'] = float(requests_per_minute.max()) if not requests_per_minute.empty else 0
        metrics['min_requests_per_minute'] = float(requests_per_minute.min()) if not requests_per_minute.empty else 0
        metrics['std_requests_per_minute'] = float(requests_per_minute.std()) if not requests_per_minute.empty else 0

        if metrics.get('mean_requests_per_minute', 0) > 0:
            metrics['cv_request_rate'] = metrics.get('std_requests_per_minute', 0) / metrics['mean_requests_per_minute']
            metrics['peak_to_average_ratio'] = metrics.get('max_requests_per_minute', 0) / metrics['mean_requests_per_minute']

        mean_rpm = metrics.get('mean_requests_per_minute', 0)
        std_rpm = metrics.get('std_requests_per_minute', 0)
        if std_rpm > 0:
            metrics['burst_score'] = int(requests_per_minute[requests_per_minute > (mean_rpm + 2 * std_rpm)].count())
        else:
            metrics['burst_score'] = 0

    if 'data.status_code' in df_entity.columns:
        status_codes = pd.to_numeric(df_entity['data.status_code'], errors='coerce').dropna()
        metrics['pct_2xx_responses'] = (status_codes.between(200, 299).sum() / total_requests) * 100
        metrics['pct_3xx_responses'] = (status_codes.between(300, 399).sum() / total_requests) * 100
        metrics['pct_4xx_responses'] = (status_codes.between(400, 499).sum() / total_requests) * 100
        metrics['pct_5xx_responses'] = (status_codes.between(500, 599).sum() / total_requests) * 100
        metrics['error_rate'] = (metrics['pct_4xx_responses'] + metrics['pct_5xx_responses'])
        metrics['critical_error_rate'] = metrics['pct_5xx_responses']
        metrics['status_code_entropy'] = calculate_entropy(status_codes)
        metrics['unique_status_codes'] = status_codes.nunique()

    if 'data.response_time' in df_entity.columns:
        response_times = pd.to_numeric(df_entity['data.response_time'], errors='coerce').dropna()
        if not response_times.empty:
            metrics['mean_response_time'] = response_times.mean()
            metrics['std_response_time'] = response_times.std()
            metrics['p50_response_time'] = response_times.quantile(0.50)
            metrics['p75_response_time'] = response_times.quantile(0.75)
            metrics['p90_response_time'] = response_times.quantile(0.90)
            metrics['p95_response_time'] = response_times.quantile(0.95)
            metrics['p99_response_time'] = response_times.quantile(0.99)
            metrics['pct_slow_requests'] = (response_times > 1).sum() / total_requests * 100
            metrics['pct_very_slow_requests'] = (response_times > 5).sum() / total_requests * 100

    if 'data.srcip' in df_entity.columns:
        ips = df_entity['data.srcip'].dropna()
        metrics['unique_source_ips'] = ips.nunique()
        if metrics['unique_source_ips'] > 0:
            metrics['mean_requests_per_ip'] = total_requests / metrics['unique_source_ips']

        ip_counts = ips.value_counts()
        metrics['max_requests_single_ip'] = int(ip_counts.max()) if not ip_counts.empty else 0
        metrics['gini_ip_distribution'] = calculate_gini(ip_counts)

        top_10_pct_ip_count = int(np.ceil(0.1 * metrics.get('unique_source_ips', 0)))
        metrics['ip_concentration_top10pct'] = ip_counts.head(top_10_pct_ip_count).sum() / total_requests

    if 'data.operator_or_user_id' in df_entity.columns:
        metrics['unique_operators'] = df_entity['data.operator_or_user_id'].nunique()
        metrics['unique_accounts'] = metrics['unique_operators']
        metrics['account_diversity_ratio'] = metrics.get('unique_accounts', 0) / total_requests

    if 'data.api_module' in df_entity.columns:
        api_modules = df_entity['data.api_module'].dropna()
        metrics['unique_api_modules'] = api_modules.nunique()
        metrics['module_entropy'] = calculate_entropy(api_modules)
        if not api_modules.empty:
            metrics['top_module_percentage'] = (api_modules.value_counts().max() / total_requests) * 100
        metrics['module_switching_frequency'] = (api_modules != api_modules.shift()).sum() / total_requests

    if 'data.route_uri' in df_entity.columns:
        routes = df_entity['data.route_uri'].dropna()
        metrics['unique_routes'] = routes.nunique()
        metrics['route_entropy'] = calculate_entropy(routes)
        if not routes.empty:
            metrics['top5_routes_percentage'] = (routes.value_counts().nlargest(5).sum() / total_requests) * 100

    if 'data.size' in df_entity.columns:
        response_sizes = pd.to_numeric(df_entity['data.size'], errors='coerce').dropna()
        if not response_sizes.empty:
            metrics['mean_response_size'] = response_sizes.mean()
            metrics['std_response_size'] = response_sizes.std()
            metrics['max_response_size'] = response_sizes.max()
            metrics['min_response_size'] = response_sizes.min()

    if 'data.browser' in df_entity.columns:
        user_agents = df_entity['data.browser'].dropna()
        metrics['unique_user_agents'] = user_agents.nunique()
        metrics['user_agent_entropy'] = calculate_entropy(user_agents)
        metrics['bot_like_ua_percentage'] = user_agents.str.contains('bot|crawler', case=False, na=False).sum() / total_requests * 100

    if 'data.method' in df_entity.columns:
        methods = df_entity['data.method'].dropna()
        metrics['unique_http_methods'] = methods.nunique()
        metrics['get_request_ratio'] = (methods == 'GET').sum() / total_requests
        metrics['post_request_ratio'] = (methods == 'POST').sum() / total_requests

    if 'data.protocol_version' in df_entity.columns:
        metrics['http11_ratio'] = (df_entity['data.protocol_version'] == 'HTTP/1.1').sum() / total_requests

    return clean_metrics(metrics)

def save_metrics_to_index(client, entity_id, timestamp, metrics, window_minutes):
    target_index = f"metrics-l1-{window_minutes}m"
    doc_id = f"{entity_id}_{timestamp}"

    document = {
        "@timestamp": timestamp,
        "entity_id": entity_id,
        "metrics": metrics,
        "time_window_minutes": window_minutes
    }

    try:
        client.index(index=target_index, body=document, id=doc_id)
    except Exception:
        pass
