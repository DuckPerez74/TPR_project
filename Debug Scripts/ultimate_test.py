# --- ultimate_test.py ---
# O teste final que contorna o problema de mapeamento de data.

from opensearchpy import OpenSearch
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
ES_HOSTS = ["https://100.125.228.80:9200"]
ES_USER = "admin"
ES_PASSWORD = "SecretPassword" # A tua password real
INDEX_PATTERN = "wazuh-alerts-4.x-*"

def main():
    print(f"--- INICIANDO O TESTE FINAL (WILDCARD) ---")
    
    try:
        client = OpenSearch(
            hosts=ES_HOSTS,
            http_auth=(ES_USER, ES_PASSWORD),
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        if not client.ping(): raise ConnectionError("Falha na autenticação.")
        print("Conectado e autenticado com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao conectar: {e}")
        return

    # A data que queremos, como texto
    date_string_to_find = "2025-09-01"
    print(f"A procurar por logs onde o timestamp começa com '{date_string_to_find}'...")

    # --- A NOVA QUERY (A MÁGICA ESTÁ AQUI) ---
    # Usamos um 'wildcard' para procurar o texto da data, em vez de um 'range'
    query = {
        "query": {
            "wildcard": {
                # Usamos .keyword para garantir que procuramos no texto exato
                "timestamp.keyword": {
                    "value": f"{date_string_to_find}*"
                }
            }
        }
    }

    try:
        response = client.count(index=INDEX_PATTERN, body=query)
        count = response.get('count', 0)
        
        print("\n--- RESULTADO DO TESTE ---")
        if count > 0:
            print(f"VITÓRIA ABSOLUTA! Foram encontrados {count} logs para o dia {date_string_to_find}.")
            print("\nCONCLUSÃO: O problema é o mapeamento do campo de data. A pesquisa por texto (wildcard) funciona!")
        else:
            print(f"FALHA FINAL. 0 logs encontrados.")
            print("\nCONCLUSÃO: Se nem a pesquisa por texto funciona, os dados estão inacessíveis via API por um motivo mais profundo.")

    except Exception as e:
        print(f"\nOcorreu um erro durante a busca: {e}")

    print("\n--- FIM DO TESTE ---")


if __name__ == "__main__":
    main()