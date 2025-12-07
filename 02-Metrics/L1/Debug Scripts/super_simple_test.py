# --- super_simple_test.py ---
# O teste mais simples e direto possível.

from datetime import datetime, timezone
from opensearchpy import OpenSearch
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
ES_HOSTS = ["https://100.125.228.80:9200"]
ES_USER = "admin"
ES_PASSWORD = "SecretPassword" # A tua password real

# --- O NOME CORRETO DO ÍNDICE ---
INDEX_PATTERN = "wazuh-alerts-4.x-*" # <--- A CORREÇÃO CRÍTICA ESTÁ AQUI

def main():
    print(f"--- INICIANDO O TESTE FINAL E MAIS SIMPLES ---")
    print(f"A procurar no padrão de índice: '{INDEX_PATTERN}'")

    try:
        client = OpenSearch(
            hosts=ES_HOSTS,
            http_auth=(ES_USER, ES_PASSWORD),
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        if not client.ping():
            raise ConnectionError("Falha na autenticação.")
        print("Conectado e autenticado com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao conectar: {e}")
        return

    # Vamos procurar no dia 1 de Setembro inteiro para garantir que apanhamos algo
    start_time = datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2025, 9, 2, 0, 0, 0, tzinfo=timezone.utc) # O dia seguinte à meia-noite
    
    print(f"A procurar por QUALQUER log entre {start_time} e {end_time}...")

    # Query para contar todos os documentos nesse intervalo
    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": start_time.isoformat(),
                    "lt": end_time.isoformat()
                }
            }
        }
    }

    try:
        response = client.count(index=INDEX_PATTERN, body=query)
        count = response.get('count', 0)
        
        print("\n--- RESULTADO DO TESTE ---")
        if count > 0:
            print(f"VITÓRIA! Foram encontrados {count} logs.")
            print("\nCONCLUSÃO: O problema era o nome do índice. O script principal vai funcionar se usarmos este nome.")
        else:
            print(f"FALHA INEXPLICÁVEL. Foram encontrados 0 logs.")
            print("\nCONCLUSÃO: Se isto falhar, o problema não pode ser resolvido por um script Python. Está relacionado com a forma como os dados estão guardados no servidor.")

    except Exception as e:
        print(f"\nOcorreu um erro durante a busca: {e}")

    print("\n--- FIM DO TESTE ---")


if __name__ == "__main__":
    main()