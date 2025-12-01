import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta, timezone
import warnings
from opensearchpy import OpenSearch

# Ignorar avisos de certificados SSL (apenas para desenvolvimento)
warnings.filterwarnings('ignore', 'Unverified HTTPS request')

# --- PASSO 0: CONFIGURAÇÃO ---
# Detalhes de conexão ao teu Wazuh Indexer (Elasticsearch/OpenSearch)
ES_HOSTS = ["https://100.125.228.80:9200"]
ES_USER = "admin"
ES_PASSWORD = "SecretPassword" 
INDEX_PATTERN = "wazuh-alerts-4.x-*" 



# O campo que identifica a empresa nos teus logs
COMPANY_ID_FIELD = "data.entities"


def calculate_entropy(series):
    """Calcula a Entropia de Shannon para uma série de dados."""
    if series.empty:
        return 0
    # Calcula a frequência de cada valor único
    counts = series.value_counts()
    # Calcula a probabilidade de cada valor
    probs = counts / counts.sum()
    # Calcula a entropia
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def calculate_gini(series):
    """Calcula o Coeficiente de Gini para uma série de contagens."""
    if series.empty or series.sum() == 0:
        return 0
    # Ordena os valores
    sorted_series = series.sort_values().to_numpy()
    n = len(sorted_series)
    cumx = np.cumsum(sorted_series)
    # Fórmula do coeficiente de Gini
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def fetch_data_from_wazuh(es_client, start_time, end_time):
    """Busca TODOS os logs no período, sem o filtro 'exists' no lado do servidor."""
    query = {
        "query": {
            "range": {
                "timestamp": {
                    "gte": start_time.isoformat(),
                    "lt": end_time.isoformat()
                }
            }
        }
    }
    
    print(f"A procurar TODOS os logs entre {start_time} e {end_time}...")
    try:
        response = es_client.search(index="wazuh-alerts-*", body=query, size=10000, scroll="2m")
        scroll_id = response.get('_scroll_id')
        hits = response['hits']['hits']
        
        while scroll_id and len(response['hits']['hits']) > 0:
            response = es_client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get('_scroll_id')
            hits.extend(response['hits']['hits'])

        source_hits = [hit['_source'] for hit in hits]
        print(f"Encontrados {len(source_hits)} logs no total neste período.")
        return source_hits
    except Exception as e:
        print(f"Ocorreu um erro ao buscar os dados: {e}")
        return []

# --- Funções Auxiliares para Cálculos ---

def calculate_l1_metrics(df_company, time_window_minutes):
    """
    Calcula TODAS as métricas L1 para o DataFrame de uma única empresa.
    NOTA: Os nomes dos campos ('data.status_code', 'data.srcip', etc.) são baseados no teu JSON.
          Ajusta-os se necessário para outros logs.
    """
    if df_company.empty:
        return {}

    metrics = {}
    total_requests = len(df_company)
    metrics['total_requests'] = total_requests

    # --- Grupo: Temporal Volume ---
    if time_window_minutes > 0:
        metrics['mean_requests_per_minute'] = total_requests / time_window_minutes
        
        # Converte o timestamp para datetime se ainda não for
        if not pd.api.types.is_datetime64_any_dtype(df_company['@timestamp']):
            df_company['@timestamp'] = pd.to_datetime(df_company['@timestamp'])

        # Agrupar por minuto para métricas mais granulares
        requests_per_minute = df_company.set_index('@timestamp').resample('1min').size()
        
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

    # --- Grupo: Status Distribution ---
    # Campo: 'data.status_code'
    if 'data.status_code' in df_company.columns:
        status_codes = pd.to_numeric(df_company['data.status_code'], errors='coerce').dropna()
        metrics['pct_2xx_responses'] = (status_codes.between(200, 299).sum() / total_requests) * 100
        metrics['pct_3xx_responses'] = (status_codes.between(300, 399).sum() / total_requests) * 100
        metrics['pct_4xx_responses'] = (status_codes.between(400, 499).sum() / total_requests) * 100
        metrics['pct_5xx_responses'] = (status_codes.between(500, 599).sum() / total_requests) * 100
        metrics['error_rate'] = (metrics['pct_4xx_responses'] + metrics['pct_5xx_responses'])
        metrics['critical_error_rate'] = metrics['pct_5xx_responses']
        metrics['status_code_entropy'] = calculate_entropy(status_codes)
        metrics['unique_status_codes'] = status_codes.nunique()

    # --- Grupo: Performance ---
    # Campo: 'data.response_time' (em segundos)
    if 'data.response_time' in df_company.columns:
        response_times = pd.to_numeric(df_company['data.response_time'], errors='coerce').dropna()
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
            
    # --- Grupo: Entity Profiling ---
    # Campos: 'data.srcip', 'data.operator_or_user_id'
    if 'data.srcip' in df_company.columns:
        ips = df_company['data.srcip'].dropna()
        metrics['unique_source_ips'] = ips.nunique()
        if metrics['unique_source_ips'] > 0:
            metrics['mean_requests_per_ip'] = total_requests / metrics['unique_source_ips']
        
        ip_counts = ips.value_counts()
        metrics['max_requests_single_ip'] = int(ip_counts.max()) if not ip_counts.empty else 0
        metrics['gini_ip_distribution'] = calculate_gini(ip_counts)
        
        top_10_pct_ip_count = int(np.ceil(0.1 * metrics.get('unique_source_ips', 0)))
        metrics['ip_concentration_top10pct'] = ip_counts.head(top_10_pct_ip_count).sum() / total_requests
        
    if 'data.operator_or_user_id' in df_company.columns:
        metrics['unique_operators'] = df_company['data.operator_or_user_id'].nunique()
        # A métrica 'unique_accounts' parece ser a mesma que 'unique_operators' no teu JSON
        metrics['unique_accounts'] = metrics['unique_operators']
        metrics['account_diversity_ratio'] = metrics.get('unique_accounts', 0) / total_requests

    # --- Grupo: API Usage ---
    # Campos: 'data.api_module', 'data.route_uri'
    if 'data.api_module' in df_company.columns:
        api_modules = df_company['data.api_module'].dropna()
        metrics['unique_api_modules'] = api_modules.nunique()
        metrics['module_entropy'] = calculate_entropy(api_modules)
        if not api_modules.empty:
            metrics['top_module_percentage'] = (api_modules.value_counts().max() / total_requests) * 100
        metrics['module_switching_frequency'] = (api_modules != api_modules.shift()).sum() / total_requests
    
    if 'data.route_uri' in df_company.columns:
        routes = df_company['data.route_uri'].dropna()
        metrics['unique_routes'] = routes.nunique()
        metrics['route_entropy'] = calculate_entropy(routes)
        if not routes.empty:
            metrics['top5_routes_percentage'] = (routes.value_counts().nlargest(5).sum() / total_requests) * 100
        
    # --- Grupo: Payload ---
    # Campo: 'data.size' (bytes)
    if 'data.size' in df_company.columns:
        response_sizes = pd.to_numeric(df_company['data.size'], errors='coerce').dropna()
        if not response_sizes.empty:
            metrics['mean_response_size'] = response_sizes.mean()
            metrics['std_response_size'] = response_sizes.std()
            metrics['max_response_size'] = response_sizes.max()
            metrics['min_response_size'] = response_sizes.min()

    # --- Grupo: Client Diversity ---
    # Campo: 'data.browser' (User-Agent)
    if 'data.browser' in df_company.columns:
        user_agents = df_company['data.browser'].dropna()
        metrics['unique_user_agents'] = user_agents.nunique()
        metrics['user_agent_entropy'] = calculate_entropy(user_agents)
        metrics['bot_like_ua_percentage'] = user_agents.str.contains('bot|crawler', case=False, na=False).sum() / total_requests * 100

    # --- Grupo: Request Patterns & Protocol ---
    # Campos: 'data.method', 'data.protocol_version'
    if 'data.method' in df_company.columns:
        methods = df_company['data.method'].dropna()
        metrics['unique_http_methods'] = methods.nunique()
        metrics['get_request_ratio'] = (methods == 'GET').sum() / total_requests
        metrics['post_request_ratio'] = (methods == 'POST').sum() / total_requests
    
    if 'data.protocol_version' in df_company.columns:
        metrics['http11_ratio'] = (df_company['data.protocol_version'] == 'HTTP/1.1').sum() / total_requests
        
    return metrics

def load_metrics_to_wazuh(es_client, company_id, timestamp, metrics, target_index):
    """
    Envia o documento de métricas calculado para um novo índice no Wazuh.
    """
    # -------------
    #print(">>> ESTOU A ENTRAR NA FUNCAO 1 <<<") 
    # ---------------------

    # Converter tudo o que for NaN ou Infinito para 0.
    clean_metrics = {}
    for key, value in metrics.items():
        # Verifica se é um número (float ou int)
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                clean_metrics[key] = 0
            else:
                clean_metrics[key] = value
        else:
            clean_metrics[key] = value # Se for string/outro, mantém
    # -------------------------------

    document = {
        "@timestamp": timestamp,
        "company_id": company_id,
        "metrics": clean_metrics,
        "time_window_minutes": metrics.get('time_window_minutes', 0)
    }
    
    try:
        es_client.index(index=target_index, body=document)
        print(f"Métricas para a empresa '{company_id}' enviadas para o índice '{target_index}'.")
    except Exception as e:
        #Debug...
        print(f"ERRO CRÍTICO ao enviar '{company_id}': {e}")
        print(f"Dados que falharam: {clean_metrics}")


def main(time_window_minutes, start_date_str, end_date_str):
    print(f"--- A iniciar processo para janelas de {time_window_minutes} minutos ---")
    
    try:
        client = OpenSearch(
            hosts=ES_HOSTS,
            http_auth=(ES_USER, ES_PASSWORD),
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        if not client.ping(): raise ConnectionError("Falha na autenticação.")
        print("Conectado e autenticado no Wazuh Indexer com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao conectar: {e}")
        return

    start_date = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)
    end_date = datetime.fromisoformat(end_date_str).replace(tzinfo=timezone.utc)
    print(f"Período de análise definido de {start_date} até {end_date} (em UTC).")

    current_time = start_date
    while current_time < end_date:
        chunk_start_time = current_time
        chunk_end_time = current_time + timedelta(minutes=time_window_minutes)
        if chunk_end_time > end_date: chunk_end_time = end_date
        
        raw_logs = fetch_data_from_wazuh(client, chunk_start_time, chunk_end_time)
        
        if raw_logs:
            df = pd.json_normalize(raw_logs, sep='.')
            
            if COMPANY_ID_FIELD in df.columns:
                df_filtered = df.dropna(subset=[COMPANY_ID_FIELD])
                print(f"Dos {len(df)} logs, {len(df_filtered)} contêm o campo '{COMPANY_ID_FIELD}' e serão processados.")

                if not df_filtered.empty:
                    # Converte o timestamp antes de agrupar
                    if 'timestamp' in df_filtered.columns:
                        df_filtered['@timestamp'] = pd.to_datetime(df_filtered['timestamp'])
                    elif '@timestamp' in df.columns:
                        df['@timestamp'] = pd.to_datetime(df['@timestamp'])

                    grouped_by_company = df_filtered.groupby(COMPANY_ID_FIELD)
                    print(f"Dados agrupados. Encontradas {len(grouped_by_company)} entidades únicas.")
                    
                    target_index = f"metrics-l1-{time_window_minutes}m"
                    for company_id, company_df in grouped_by_company:

                        # Se o ID for um traço ou vazio, salta para o próximo e ignora este
                        if str(company_id).strip() in ["-", ""]:
                            continue
                        print(f"Processando entidade: {company_id}...")
                        calculated_metrics = calculate_l1_metrics(company_df, time_window_minutes)
                        calculated_metrics['time_window_minutes'] = time_window_minutes
                        load_metrics_to_wazuh(client, company_id, chunk_end_time, calculated_metrics, target_index)
            else:
                print(f"AVISO: Nenhum dos {len(df)} logs neste período continha o campo '{COMPANY_ID_FIELD}'.")
        
        current_time += timedelta(minutes=time_window_minutes)

    print(f"\n--- Processo histórico concluído! ---")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa logs históricos do Wazuh e calcula métricas L1.")
    
    parser.add_argument(
        '--minutes', type=int, required=True,
        help='A janela de tempo em minutos para cada fatia de análise (ex: 60).'
    )
    parser.add_argument(
        '--start-date', type=str, required=True,
        help='A data/hora de início no formato ISO (ex: "2025-08-01T00:00:00").'
    )
    parser.add_argument(
        '--end-date', type=str, required=True,
        help='A data/hora de fim no formato ISO (ex: "2025-08-01T23:59:59").'
    )
    
    args = parser.parse_args()
    
    main(
        time_window_minutes=args.minutes,
        start_date_str=args.start_date,
        end_date_str=args.end_date
    )