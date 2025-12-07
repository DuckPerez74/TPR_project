# --- test_query.py ---

import argparse
from datetime import datetime, timezone
from opensearchpy import OpenSearch
import warnings
import json

# Ignorar avisos
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
ES_HOSTS = ["https://100.125.228.80:9200"]
ES_USER = "admin"
ES_PASSWORD = "SecretPassword" # Usa a tua password real

# --- O CAMPO QUE VAMOS TESTAR ---
FIELD_TO_TEST = "data.status_code" # Um campo comum que existe no teu JSON de exemplo

def main(start_date_str, end_date_str):
    print(f"--- A INICIAR TESTE DE QUERY SIMPLES ---")
    print(f"Campo a procurar: '{FIELD_TO_TEST}'")

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

    # Corrigir o fuso horário (timezone)
    start_date = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)
    end_date = datetime.fromisoformat(end_date_str).replace(tzinfo=timezone.utc)
    
    print(f"A procurar no período de {start_date} a {end_date} (UTC)...")

    # Query para encontrar documentos que TENHAM o campo de teste
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_date.isoformat(),
                                "lt": end_date.isoformat()
                            }
                        }
                    },
                    {
                        "exists": {
                            "field": FIELD_TO_TEST
                        }
                    }
                ]
            }
        }
    }

    try:
        # Usar a função 'count' que é mais simples e direta para este teste
        response = client.count(index="wazuh-alerts-*", body=query)
        count = response.get('count', 0)
        
        print("\n--- RESULTADO DO TESTE ---")
        if count > 0:
            print(f"SUCESSO! Foram encontrados {count} logs com o campo '{FIELD_TO_TEST}'.")
            print("\nCONCLUSÃO: O problema está especificamente no campo 'data.entities'. O script e a conexão estão corretos.")
        else:
            print(f"FALHA. Foram encontrados 0 logs com o campo '{FIELD_TO_TEST}'.")
            print("\nCONCLUSÃO: O problema é mais profundo, relacionado com a forma como os dados estão guardados ou indexados.")

    except Exception as e:
        print(f"\nOcorreu um erro durante a busca: {e}")

    print("\n--- FIM DO TESTE ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa uma query simples no Wazuh.")
    parser.add_argument('--start-date', type=str, required=True, help='Data de início (ex: "2025-09-01T00:00:00").')
    parser.add_argument('--end-date', type=str, required=True, help='Data de fim (ex: "2025-09-01T03:00:00").')
    args = parser.parse_args()
    main(args.start_date, args.end_date)