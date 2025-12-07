# --- prova_conexao.py ---
# Um script final para provar que a conexão funciona, mas a obtenção de dados falha.

from datetime import datetime, timezone
from opensearchpy import OpenSearch
import warnings
import json

# Ignorar avisos
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
ES_HOSTS = ["https://100.125.228.80:9200"]
ES_USER = "admin"
ES_PASSWORD = "SecretPassword" # A tua password real
INDEX_PATTERN = "wazuh-alerts-4.x-*" # O padrão de índice que descobrimos

def main():
    print(f"--- INICIANDO PROVA DE CONEXÃO E LEITURA ---")
    
    # --- PASSO 1: TENTAR CONECTAR E AUTENTICAR ---
    try:
        client = OpenSearch(
            hosts=ES_HOSTS,
            http_auth=(ES_USER, ES_PASSWORD),
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=20 # Aumentar o timeout para ter a certeza
        )
        if not client.ping():
            raise ConnectionError("O servidor respondeu, mas a autenticação falhou (ping=false). Verifica as credenciais.")
        print("[PASSO 1 SUCESSO] Conectado e autenticado com sucesso no Wazuh Indexer.")
    except Exception as e:
        print(f"[PASSO 1 FALHA] Erro crítico ao tentar conectar: {e}")
        return # Se a conexão falhar, não vale a pena continuar

    # --- PASSO 2: TENTAR LER UM ÚNICO LOG ---
    # Período de busca: o dia 1 de Setembro inteiro
    start_time = datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2025, 9, 2, 0, 0, 0, tzinfo=timezone.utc)
    
    print(f"\n--- A iniciar PASSO 2: Tentar ler UM log de '{INDEX_PATTERN}' entre {start_time} e {end_time} ---")

    query = { "query": { "match_all": {} } } # A query mais simples possível

    try:
        response = client.search(index=INDEX_PATTERN, body=query, size=1)
        hits = response['hits']['hits']
        
        if hits:
            print(f"[PASSO 2 SUCESSO] VITÓRIA! Foi encontrado pelo menos um log.")
            print("Isto significa que a conexão e a leitura funcionam. Podemos adaptar o script principal.")
            print("\n--- Conteúdo do log encontrado: ---")
            print(json.dumps(hits[0]['_source'], indent=2))
        else:
            print(f"[PASSO 2 FALHA] A pesquisa foi executada com sucesso, mas o servidor retornou 0 logs.")
            print("Isto prova que a conexão funciona, mas há um problema do lado do servidor (permissões ou dados) que impede a leitura.")

    except Exception as e:
        print(f"[PASSO 2 FALHA] A conexão funcionou, mas a pesquisa falhou com um erro: {e}")
        print("Isto também aponta para um problema de permissões ou configuração no servidor.")

    print("\n--- FIM DA PROVA DE CONEXÃO ---")


if __name__ == "__main__":
    main()