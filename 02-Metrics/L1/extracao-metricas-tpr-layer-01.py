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

    # O ID vai ser algo como: "436_2025-09-01T08:00:00"
    # Assim, se correres de novo, o ID é igual e ele atualiza em vez de duplicar.
    doc_id = f"{company_id}_{timestamp}"
    # -------------------------------

    document = {
        "@timestamp": timestamp,
        "company_id": company_id,
        "metrics": clean_metrics,
        "time_window_minutes": metrics.get('time_window_minutes', 0)
    }
    
    try:
        es_client.index(index=target_index, body=document, id=doc_id)
        print(f"Métricas para a empresa '{company_id}' enviadas para o índice '{target_index}'.")
    except Exception as e:
        #Debug...
        print(f"ERRO CRÍTICO ao enviar '{company_id}': {e}")
        print(f"Dados que falharam: {clean_metrics}")


def main(analysis_window_minutes, start_date_str, end_date_str):
    """
    analysis_window_minutes: O tamanho da janela que queres CALCULAR (ex: 10 min)
    """
    print(f"--- A iniciar processo. Janela de Análise: {analysis_window_minutes} min ---")
    
    # 1. Configuração da Janela de FETCH (busca)
    # A ideia é buscar sempre blocos de 60 minutos (1 hora) para reduzir as queries.
    # Se a janela de análise for maior que 60 (ex: 120 min), usamos a própria janela de análise para buscar.
    FETCH_WINDOW_MINUTES = max(60, analysis_window_minutes)
    
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
    print(f"Estratégia: Buscar dados a cada {FETCH_WINDOW_MINUTES} min e fatiar a cada {analysis_window_minutes} min.")

    # --- LOOP EXTERNO: Controla o FETCH (Busca de dados grossa) ---
    current_fetch_time = start_date
    while current_fetch_time < end_date:
        fetch_end_time = current_fetch_time + timedelta(minutes=FETCH_WINDOW_MINUTES)
        if fetch_end_time > end_date: fetch_end_time = end_date
        
        # Busca 1 hora de dados (ou o valor de FETCH_WINDOW_MINUTES)
        raw_logs = fetch_data_from_wazuh(client, current_fetch_time, fetch_end_time)
        
        # Se houver dados, convertemos para DataFrame UMA VEZ
        df_large_chunk = pd.DataFrame()
        if raw_logs:
            df_large_chunk = pd.json_normalize(raw_logs, sep='.')
            
            # Normalização de timestamps para garantir que o Pandas consegue filtrar
            if 'timestamp' in df_large_chunk.columns:
                df_large_chunk['@timestamp'] = pd.to_datetime(df_large_chunk['timestamp'])
            elif '@timestamp' in df_large_chunk.columns:
                df_large_chunk['@timestamp'] = pd.to_datetime(df_large_chunk['@timestamp'])
            
            # Garante que o timestamp tem timezone UTC para bater certo com as tuas datas
            if not df_large_chunk.empty and df_large_chunk['@timestamp'].dt.tz is None:
                 df_large_chunk['@timestamp'] = df_large_chunk['@timestamp'].dt.tz_localize('UTC')

        # --- LOOP INTERNO: Controla o SLICING (Fatiamento para análise) ---
        # Agora vamos percorrer a "Fetch Window" em pedacinhos de "Analysis Window"
        current_slice_time = current_fetch_time
        
        while current_slice_time < fetch_end_time:
            slice_end_time = current_slice_time + timedelta(minutes=analysis_window_minutes)
            
            # Se por acaso o slice passar do tempo total do script, cortamos
            if slice_end_time > end_date: slice_end_time = end_date

            # Se não houver logs no bloco grande, não há nada para processar neste slice
            if df_large_chunk.empty:
                print(f"Sem dados para o intervalo {current_slice_time} - {slice_end_time}")
            else:
                # AQUI ESTÁ A MAGIA: Filtragem em Memória (Pandas) em vez de query ao Wazuh
                # Filtramos o DataFrame grande para pegar apenas os registos deste intervalo pequeno
                mask = (df_large_chunk['@timestamp'] >= current_slice_time) & (df_large_chunk['@timestamp'] < slice_end_time)
                df_slice = df_large_chunk.loc[mask].copy()

                if not df_slice.empty and COMPANY_ID_FIELD in df_slice.columns:
                    df_filtered = df_slice.dropna(subset=[COMPANY_ID_FIELD])
                    
                    if not df_filtered.empty:
                        print(f"Processando slice {current_slice_time} a {slice_end_time}: {len(df_filtered)} logs encontrados.")
                        grouped_by_company = df_filtered.groupby(COMPANY_ID_FIELD)
                        
                        target_index = f"metrics-l1-{analysis_window_minutes}m"
                        
                        for company_id, company_df in grouped_by_company:
                            if str(company_id).strip() in ["-", ""]:
                                continue
                            
                            # Calcula métricas para este pedaço pequeno
                            calculated_metrics = calculate_l1_metrics(company_df, analysis_window_minutes)
                            calculated_metrics['time_window_minutes'] = analysis_window_minutes
                            
                            # Envia para o Wazuh
                            load_metrics_to_wazuh(client, company_id, slice_end_time, calculated_metrics, target_index)
                    else:
                        print(f"Slice {current_slice_time}: Logs existem mas sem '{COMPANY_ID_FIELD}'.")
                else:
                    # Opcional: print(f"Nenhum log encontrado especificamente entre {current_slice_time} e {slice_end_time}")
                    pass

            # Avança o loop interno
            current_slice_time += timedelta(minutes=analysis_window_minutes)

        # Avança o loop externo
        current_fetch_time = fetch_end_time

    print(f"\n--- Processo histórico concluído! ---")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa logs históricos do Wazuh com otimização de fetch.")
    
    parser.add_argument(
        '--minutes', type=int, required=True,
        help='A janela de tempo em minutos para o CÁLCULO das métricas (ex: 10).'
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
        analysis_window_minutes=args.minutes,
        start_date_str=args.start_date,
        end_date_str=args.end_date
    )