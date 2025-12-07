from opensearchpy import OpenSearch
import warnings

# --- CONFIGURAÇÃO (A mesma de sempre) ---
ES_HOSTS = ["https://100.125.228.80:9200"]
ES_USER = "admin"
ES_PASSWORD = "SecretPassword" 
warnings.filterwarnings('ignore')

client = OpenSearch(
    hosts=ES_HOSTS,
    http_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

def get_boundary_log(order):
    """Obtém o primeiro ou último log baseado na ordem (asc ou desc)."""
    try:
        response = client.search(
            index="wazuh-alerts-*", 
            body={
                "size": 1,
                "sort": [
                    {"timestamp": {"order": order}}  # Ordena pelo tempo
                ],
                "_source": ["timestamp"] # Só queremos saber a data
            }
        )
        hits = response['hits']['hits']
        if hits:
            return hits[0]['_source'].get('timestamp')
        return None
    except Exception as e:
        print(f"Erro: {e}")
        return None

print("--- A procurar os limites temporais dos teus dados ---")

# 1. Procurar o mais antigo (Ascendente)
oldest = get_boundary_log("asc")
if oldest:
    print(f"\n📅 DATA MAIS ANTIGA (Início): {oldest}")
    print("   -> Usa esta data no --start-date")
else:
    print("Não foi possível encontrar a data mais antiga.")

# 2. Procurar o mais recente (Descendente)
newest = get_boundary_log("desc")
if newest:
    print(f"\n📅 DATA MAIS RECENTE (Fim):    {newest}")
    print("   -> Usa esta data no --end-date")
else:
    print("Não foi possível encontrar a data mais recente.")

print("\n------------------------------------------------")