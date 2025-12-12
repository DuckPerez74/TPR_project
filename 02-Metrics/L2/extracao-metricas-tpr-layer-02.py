import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta, timezone
import warnings
from opensearchpy import OpenSearch
import re
from datetime import timedelta
import geoip2.database
from math import radians, cos, sin, asin, sqrt

# Path para o ficheiro GeoLite2
GEOIP_DB_PATH = "GeoLite2-City.mmdb" 


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

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calcula a distância (em km) entre dois pontos geográficos (Lat/Lon).
    """
    # Converter graus para radianos
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Fórmula de Haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Raio da Terra em km
    return c * r


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

# --- Funções auxiliares para cálculos adicionais camadas L2 ---

def calculate_l2_full_features(df_logs):
    """
    Cálculo exaustivo das features L2 baseadas no PDF.
    Recebe os logs brutos (df_logs) de uma empresa numa janela de tempo.
    """
    metrics = {}
    if df_logs.empty: return metrics

    # --- SETUP: Garantir colunas essenciais ---
    # Cria colunas auxiliares para evitar erros se não existirem
    if 'data.url' not in df_logs.columns: df_logs['data.url'] = ""
    if 'data.method' not in df_logs.columns: df_logs['data.method'] = ""
    if 'data.response' not in df_logs.columns: df_logs['data.response'] = 0
    
    # Converter URLs para string para usar regex
    urls = df_logs['data.url'].astype(str).fillna("")
    methods = df_logs['data.method'].astype(str).fillna("")
    
    # Tentar converter status code para número
    status_codes = pd.to_numeric(df_logs['data.response'], errors='coerce').fillna(0)

    # ==============================================================================
    # GRUPO: ACCESS PATTERNS & SECURITY (PDF Pag 7, 9)
    # ==============================================================================

    # 1. privilege_endpoint_ratio
    # Rácio de acessos a endpoints administrativos
    admin_keywords = 'admin|config|setup|root|dashboard'
    admin_count = urls.str.contains(admin_keywords, case=False).sum()
    metrics['privilege_endpoint_ratio'] = admin_count / len(df_logs)

    # 2. sensitive_data_access_rate (Igual ao financial/PII)
    sensitive_keywords = 'billing|finance|salary|payments|cpf|nif|ssn|credit_card'
    sensitive_count = urls.str.contains(sensitive_keywords, case=False).sum()
    metrics['sensitive_data_access_rate'] = sensitive_count / len(df_logs)

    # 3. sequential_resource_access (Deteção de Enumeration)
    # Procura padrões onde o URL é igual mas só muda o ID final (ex: /user/101, /user/102)
    # Simplificação: Contar URLs que terminam em dígitos diferentes
    # Se extrairmos os IDs e tivermos muitos IDs únicos para o mesmo "caminho base", é enumeration.
    try:
        # Extrai o último numero do URL
        ids = urls.str.extract(r'(\d+)$')[0].dropna()
        if len(ids) > 5: # Só conta se houver pelo menos 5 acessos numéricos
            # Calcula quantos IDs únicos existem vs total de acessos
            metrics['sequential_resource_access'] = len(ids.unique()) / len(ids)
        else:
            metrics['sequential_resource_access'] = 0
    except:
        metrics['sequential_resource_access'] = 0

    # 4. cross_department_access
    # Difícil saber o departamento sem uma tabela de RH. 
    # Vamos assumir "0" por agora ou usar acessos a pastas diferentes da raiz habitual.
    metrics['cross_department_access'] = 0 # Placeholder (precisa de dados externos)

    # 5. config_modification_attempts (PDF Pag 9)
    # Métodos PUT/POST/DELETE em endpoints de config
    config_mods = (methods.isin(['PUT', 'POST', 'PATCH', 'DELETE'])) & (urls.str.contains('config|setting', case=False))
    metrics['config_modification_attempts'] = config_mods.sum()

    # 6. backup_access_indicator (PDF Pag 9)
    backup_keywords = 'backup|dump|archive|tar.gz|zip|sql'
    metrics['backup_access_indicator'] = urls.str.contains(backup_keywords, case=False).sum()

    # ==============================================================================
    # GRUPO: VOLUME ANOMALIES (PDF Pag 8)
    # ==============================================================================

    # 7. export_endpoint_usage
    export_keywords = 'export|download|report|csv|xlsx'
    metrics['export_endpoint_usage'] = urls.str.contains(export_keywords, case=False).sum()

    # 8. failed_attempts_before_success (Brute Force Indicator)
    # Padrão: Vários 4xx seguidos de um 2xx
    # Isto é complexo em pandas vetorizado, vamos simplificar:
    # Rácio de 403 (Forbidden) ou 401 (Unauthorized)
    forbidden_count = status_codes.isin([401, 403]).sum()
    metrics['auth_failure_ratio'] = forbidden_count / len(df_logs)
    
    # 9. bulk_operation_ratio
    # URLs que contêm "batch" ou "bulk"
    metrics['bulk_operation_ratio'] = urls.str.contains('batch|bulk|multi', case=False).sum()

    # ==============================================================================
    # GRUPO: USER BASELINE & CONTEXT (PDF Pag 7, 8)
    # ==============================================================================

    # 10. working_hours_deviation
    log_hours = df_logs['@timestamp'].dt.hour
    # Assume horas normais 08-19.
    metrics['working_hours_deviation'] = len(df_logs[~log_hours.between(8, 19)]) / len(df_logs)

    # 11. session_hijack_indicator (Mudança de IP na mesma sessão)
    # Se tivermos o campo 'data.srcip'
    if 'data.srcip' in df_logs.columns:
        unique_ips = df_logs['data.srcip'].nunique()
        # Se na mesma hora (sessão curta) usou 2+ IPs, é suspeito
        metrics['session_hijack_indicator'] = 1 if unique_ips > 1 else 0
    else:
        metrics['session_hijack_indicator'] = 0

    # 12. geo_location_anomaly
    # Se tivermos 'data.geoip.country_name'
    if 'data.geoip.country_name' in df_logs.columns:
        unique_countries = df_logs['data.geoip.country_name'].nunique()
        metrics['geo_location_anomaly'] = 1 if unique_countries > 1 else 0
    else:
        metrics['geo_location_anomaly'] = 0

    # ==============================================================================
    # GRUPO: VELOCITY IMPOSSIBILITY (GeoIP)
    # ==============================================================================
    
    # 13. velocity_impossibility
    # Tenta calcular viagens impossíveis se tivermos IPs e a BD GeoIP
    max_speed_detected = 0
    metrics['velocity_impossibility'] = 0 # Default

    if 'data.srcip' in df_logs.columns:
        try:
            # Carrega a base de dados (Tenta abrir apenas se existir)
            reader = geoip2.database.Reader(GEOIP_DB_PATH)
            
            # Ordenar logs por tempo para ver a sequência de viagem
            sorted_logs = df_logs.sort_values('@timestamp')[['@timestamp', 'data.srcip']].dropna()
            
            # Precisamos de pelo menos 2 logs para calcular velocidade
            if len(sorted_logs) > 1:
                prev_lat, prev_lon, prev_time = None, None, None
                
                for index, row in sorted_logs.iterrows():
                    ip = row['data.srcip']
                    time = row['@timestamp']
                    
                    # Ignorar IPs privados/locais
                    if ip.startswith(('192.168.', '10.', '172.16.', '127.')): continue
                    
                    try:
                        response = reader.city(ip)
                        curr_lat = response.location.latitude
                        curr_lon = response.location.longitude
                        
                        if prev_lat is not None:
                            # Calcular distância (km)
                            dist = haversine_distance(prev_lon, prev_lat, curr_lon, curr_lat)
                            
                            # Calcular diferença de tempo (horas)
                            time_diff = (time - prev_time).total_seconds() / 3600.0
                            
                            if time_diff > 0 and dist > 50: # Ignorar distâncias pequenas (erros de ISP)
                                speed = dist / time_diff
                                if speed > max_speed_detected:
                                    max_speed_detected = speed
                        
                        # Atualizar ponto anterior
                        prev_lat, prev_lon, prev_time = curr_lat, curr_lon, time
                        
                    except Exception:
                        continue # IP não encontrado na BD

            reader.close()
            
            # Se a velocidade for superior a 800 km/h, é "Impossível" (Viagem de avião instantânea)
            if max_speed_detected > 800:
                metrics['velocity_impossibility'] = 1 # ALERTA
            else:
                metrics['velocity_impossibility'] = 0
                
        except FileNotFoundError:
            # Se não tiveres o ficheiro .mmdb, não faz mal, mete 0
            # print("Aviso: GeoLite2-City.mmdb não encontrado. Velocity ignorada.")
            metrics['velocity_impossibility'] = 0
        except Exception as e:
            # print(f"Erro GeoIP: {e}")
            metrics['velocity_impossibility'] = 0
    
    return metrics


def get_historical_baseline(es_client, company_id, current_time, lookback_days=15):
    """
    Grupo B: Consulta o índice L1 para obter a média e desvio padrão dos últimos 15 dias.
    Retorna um dicionário com as baselines.
    """
    # Definir a janela de tempo histórica (ex: agora - 15 dias)
    end_time = current_time
    start_time = current_time - timedelta(days=lookback_days)
    
    # Query de Agregação: Pedimos ao OpenSearch para calcular as médias por nós
    query = {
        "size": 0,  # Não queremos os logs, só os números
        "query": {
            "bool": {
                "must": [
                    {"term": {"company_id": company_id}},
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_time.isoformat(),
                                "lt": end_time.isoformat()
                            }
                        }
                    }
                ]
            }
        },
        "aggs": {
            # 1. Baseline de Volume (Requests)
            "stats_requests": {
                "extended_stats": {"field": "metrics.total_requests"}
            },
            # 2. Baseline de Tamanho (Data Download)
            "stats_size": {
                "extended_stats": {"field": "metrics.mean_response_size"}
            },
            # 3. Baseline de Erros (Para detetar Scanning/Brute Force)
            "stats_errors": {
                "extended_stats": {"field": "metrics.error_rate"}
            }
        }
    }
    
    # Valores por defeito (caso seja a primeira vez que a empresa aparece)
    baselines = {
        'req_avg': 0, 'req_std': 1,
        'size_avg': 0, 'size_std': 1,
        'err_avg': 0, 'err_std': 1
    }

    try:
        # ATENÇÃO: Aqui lemos do índice L1 (metrics-l1-*), porque é lá que está o histórico!
        response = es_client.search(index="metrics-l1-*", body=query, ignore_unavailable=True)
        
        # Função auxiliar para extrair valores sem crashar
        def extract_stats(agg_name):
            stats = response.get('aggregations', {}).get(agg_name, {})
            avg = stats.get('avg')
            std = stats.get('std_deviation')
            
            # Se for None (sem dados) devolve 0 e 1
            if avg is None: avg = 0
            if std is None or std == 0: std = 1 # Evitar divisão por zero
            return avg, std

        baselines['req_avg'], baselines['req_std'] = extract_stats('stats_requests')
        baselines['size_avg'], baselines['size_std'] = extract_stats('stats_size')
        baselines['err_avg'], baselines['err_std'] = extract_stats('stats_errors')
        
    except Exception as e:
        print(f"Aviso: Não foi possível obter histórico para {company_id}: {e}")
        # Retorna os valores por defeito (neutros)
    
    return baselines

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
    Envia as métricas para o Wazuh.
    Aceita um 'doc_id' opcional para garantir que não há duplicados.
    """
    
    # -------------
    #print(">>> ESTOU A ENTRAR NA FUNCAO 1 <<<") 

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
            clean_metrics[key] = value 
    # -------------------------------

    # O ID vai ser algo como: "436_2025-09-01T08:00:00"
    doc_id = f"{company_id}_{timestamp}_L2"
    # -------------------------------

    document = {
        "@timestamp": timestamp,
        "company_id": company_id,
        "metrics": clean_metrics,
        "time_window_minutes": metrics.get('time_window_minutes', 0)
    }
    
    try:
        es_client.index(index=target_index, body=document, id=doc_id)
        #print(f"Métricas para a empresa '{company_id}' enviadas para o índice '{target_index}'.")
    except Exception as e:
        #Debug...
        print(f"ERRO CRÍTICO ao enviar '{company_id}': {e}")
        print(f"Dados que falharam: {clean_metrics}")


# --- FUNÇÃO PRINCIPAL L2 (COMPORTAMENTAL) ---

def main(time_window_minutes, start_date_str, end_date_str):
    print(f"--- INICIANDO EXTRAÇÃO L2 (Comportamental + Histórico) ---")
    print(f"Janela: {time_window_minutes} min | De: {start_date_str} Até: {end_date_str}")
    
    # 1. Configurar Cliente OpenSearch
    try:
        client = OpenSearch(
            hosts=ES_HOSTS,
            http_auth=(ES_USER, ES_PASSWORD),
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        if not client.ping(): raise ConnectionError("Falha no Ping")
        print("Conectado ao OpenSearch.")
    except Exception as e:
        print(f"ERRO CRÍTICO de Conexão: {e}")
        return

    # 2. Conversão de Datas com Timezone UTC
    # Adicionamos timezone.utc para garantir compatibilidade com o Wazuh
    start_date = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)
    end_date = datetime.fromisoformat(end_date_str).replace(tzinfo=timezone.utc)

    current_time = start_date
    
    # --- LOOP TEMPORAL (Janela a Janela) ---
    while current_time < end_date:
        chunk_start_time = current_time
        chunk_end_time = current_time + timedelta(minutes=time_window_minutes)
        
        # Garante que não passamos do limite final
        if chunk_end_time > end_date: chunk_end_time = end_date
        
        print(f"\n> Processando janela: {chunk_start_time} - {chunk_end_time}")
        
        # A. Buscar Logs Brutos (Usando a mesma função do script L1)
        raw_logs = fetch_data_from_wazuh(client, chunk_start_time, chunk_end_time)
        
        if raw_logs:
            # Converter JSON para Tabela (DataFrame)
            df = pd.json_normalize(raw_logs, sep='.')
            
            # Correção do Timestamp (importante!)
            if 'timestamp' in df.columns:
                df['@timestamp'] = pd.to_datetime(df['timestamp'])
            elif '@timestamp' in df.columns:
                df['@timestamp'] = pd.to_datetime(df['@timestamp'])

            # Verificar se temos o campo da empresa
            if COMPANY_ID_FIELD in df.columns:
                grouped_by_company = df.groupby(COMPANY_ID_FIELD)
                
                # Definir o nome do índice de destino L2
                target_index_l2 = f"metrics-l2-{time_window_minutes}m"
                
                print(f"  -> Encontradas {len(grouped_by_company)} entidades. A calcular features...")

                # --- LOOP POR EMPRESA ---
                for company_id, company_df in grouped_by_company:
                    # Ignorar IDs inválidos
                    if str(company_id).strip() in ["-", ""]: continue

                    # ----------------------------------------------------
                    # PASSO 1: Calcular Métricas L1 (Volume Atual)
                    # (Precisamos disto para comparar com o histórico)
                    # ----------------------------------------------------
                    l1_metrics = calculate_l1_metrics(company_df, time_window_minutes)
                    
                    # ----------------------------------------------------
                    # PASSO 2: Calcular L2 Contexto (Regex / Padrões)
                    # (Função calculate_l2_full_features)
                    # ----------------------------------------------------
                    l2_context = calculate_l2_full_features(company_df)
                    
                    # ----------------------------------------------------
                    # PASSO 3: Calcular L2 Baseline (Histórico 15 Dias)
                    # ----------------------------------------------------
                    # Vai ao OpenSearch buscar médias e desvios padrão antigos
                    baselines = get_historical_baseline(client, company_id, chunk_end_time, lookback_days=15)
                    
                    l2_baseline_metrics = {}
                    
                    # ----------------------------------------------------
                    # PASSO 4: Calcular Z-Scores (Desvios)
                    # Fórmula: (Valor Atual - Média Histórica) / Desvio Padrão
                    # ----------------------------------------------------
                    
                    # A. Volume de Pedidos (deviation_from_personal_baseline)
                    curr_req = l1_metrics.get('total_requests', 0)
                    l2_baseline_metrics['deviation_from_personal_baseline'] = (curr_req - baselines['req_avg']) / baselines['req_std']
                    
                    # B. Tamanho de Download (data_download_spike)
                    # Usa o 'mean_response_size' calculado no L1
                    curr_size = l1_metrics.get('mean_response_size', 0)
                    l2_baseline_metrics['data_download_spike'] = (curr_size - baselines['size_avg']) / baselines['size_std']

                    # C. Taxa de Erros (Para detetar ataques/scans)
                    curr_err = l1_metrics.get('error_rate', 0)
                    l2_baseline_metrics['error_rate_deviation'] = (curr_err - baselines['err_avg']) / baselines['err_std']

                    # Guardamos também as médias históricas no JSON (útil para debug)
                    l2_baseline_metrics['hist_avg_requests'] = baselines['req_avg']
                    l2_baseline_metrics['hist_avg_size'] = baselines['size_avg']

                    # ----------------------------------------------------
                    # PASSO 5: Juntar Tudo e Enviar
                    # ----------------------------------------------------
                    final_metrics = {}
                    final_metrics.update(l1_metrics)         # Dados L1
                    final_metrics.update(l2_context)         # Dados L2 Contexto
                    final_metrics.update(l2_baseline_metrics)# Dados L2 Histórico
                    
                    # Adiciona info da janela
                    final_metrics['time_window_minutes'] = time_window_minutes


                    
                    print("\n" + "="*50)
                    print(f"DADOS A ENVIAR PARA WAZUH. ESTES DADOS SÃO AVALIADOS PELO AUTOENCODER:\n")
                    print(f"Relatório de Teste para Empresa: {company_id}")
                    print(f"Total Pedidos (L1): {l1_metrics.get('total_requests')}")
                    print(f"Média Histórica (Baselines): {baselines.get('req_avg')}")
                    print(f"Z-Score Volume: {l2_baseline_metrics.get('deviation_from_personal_baseline')}")
                    print(f"Acessos Admin: {l2_context.get('privilege_endpoint_ratio')}")
                    print("-" * 20)
                    print(f"Viagem Impossível (GeoIP): {l2_context.get('velocity_impossibility')}") 
                    print("-" * 20)

                    # print(json.dumps(final_metrics, indent=2)) # Descomenta se quiseres ver TUDO
                    print("="*50 + "\n")
                    # -------------------------

                    # load_metrics_to_wazuh(...)

                    # Enviar para o Wazuh
                    # (A função load trata de criar o ID único com _L2 no fim)
                    load_metrics_to_wazuh(
                        es_client=client, 
                        company_id=company_id, 
                        timestamp=chunk_end_time, 
                        metrics=final_metrics, 
                        target_index=target_index_l2
                    )

            else:
                print(f"  -> AVISO: Campo '{COMPANY_ID_FIELD}' não encontrado neste chunk.")
        
        # Avançar para a próxima hora/janela
        current_time += timedelta(minutes=time_window_minutes)

    print("\n--- Processamento L2 Concluído com Sucesso! ---")

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
        time_window_minutes=args.minutes,
        start_date_str=args.start_date,
        end_date_str=args.end_date
    )