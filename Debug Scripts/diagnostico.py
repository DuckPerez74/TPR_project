from opensearchpy import OpenSearch
import json
import warnings

# --- CONFIG (A mesma de sempre) ---
ES_HOSTS = ["https://100.125.228.80:9200"]
ES_USER = "admin"
ES_PASSWORD = "SecretPassword" # <-- A TUA PASSWORD
warnings.filterwarnings('ignore')

client = OpenSearch(
    hosts=ES_HOSTS,
    http_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

# --- ALVO ESPECÍFICO: O índice que vimos na imagem ---
TARGET_INDEX = "wazuh-alerts-4.x-2025.09.01"

print(f"--- A tentar ler diretamente de: {TARGET_INDEX} ---")

try:
    # Query "Match All" - Dá-me tudo o que tiveres, apenas 1 documento
    response = client.search(
        index=TARGET_INDEX, 
        body={
            "query": { "match_all": {} },
            "size": 1
        }
    )
    
    hits = response['hits']['hits']
    if len(hits) > 0:
        log = hits[0]['_source']
        print("\nSUCESSO! Conseguimos ler um log antigo.")
        print(f"Timestamp do log: {log.get('@timestamp', 'NÃO ENCONTRADO')}")
        
        # Vamos verificar se o campo entities existe aqui
        print(f"Estrutura (Resumida): {list(log.keys())}")
        
        # Dump para veres os campos com os teus olhos
        print("\n--- JSON COMPLETO (Copia isto para analisares) ---")
        print(json.dumps(log, indent=2))
    else:
        print("ESTRANHO: O índice existe mas retornou 0 logs com match_all.")

except Exception as e:
    print(f"ERRO: {e}")