"""
Módulo para fazer previsões de cluster para novas entidades usando o modelo K-Means treinado.

Funcionalidades:
1. Fazer previsão de uma janela única
2. Fazer previsão com confiança
3. Ir buscar 7 dias de dados ao Wazuh para uma entidade específica e fazer previsões

Uso CLI (com fetch ao Wazuh):
    python predictor.py --company-id ENTITY_001 --start "2025-10-01T00:00:00" --end "2025-10-08T00:00:00"
    python predictor.py --company-id ENTITY_001 --start "2025-10-01T00:00:00" --end "2025-10-08T00:00:00" --es-user admin --es-password mypass

Uso em código:
    from predictor import KMeansPredictor
    
    predictor = KMeansPredictor()
    cluster = predictor.predict({'total_requests': 1500, ...})
    print(f"Cluster: {cluster}")
"""

from joblib import load
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Sequence
import argparse
import os
from datetime import datetime, timezone
from opensearchpy import OpenSearch
import urllib3
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIG WAZUH / OPENSEARCH ---
ES_HOSTS = os.getenv("ES_HOSTS", "https://100.125.228.80:9200").split(",")
ES_USER = os.getenv("ES_USER", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "SecretPassword")
METRICS_INDEX_PATTERN = "metrics-l1-60m*"


class KMeansPredictor:
    """
    Carrega um modelo K-Means treinado e usa-o para fazer previsões em novas entidades.
    """
    
    def __init__(
        self, 
        model_path: str = 'kmeans_model.joblib',
        scaler_path: str = 'scaler_model.joblib',
        features_path: str = 'kmeans_features.txt'
    ):
        """
        Inicializa o preditor carregando o modelo e scaler.
        
        Args:
            model_path: Caminho para o ficheiro do modelo K-Means
            scaler_path: Caminho para o ficheiro do scaler
            features_path: Caminho para o ficheiro com os nomes das features
        """
        try:
            self.kmeans = load(model_path)
            self.scaler = load(scaler_path)
            print(f"[OK] Modelo carregado de {model_path}")
            print(f"[OK] Scaler carregado de {scaler_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Ficheiros do modelo não encontrados. Certifique-se que executou o treino primeiro: {e}"
            )
        
        # Carregar nomes das features
        try:
            with open(features_path, 'r') as f:
                self.feature_cols = f.read().strip().split(',')
            print(f"[OK] Features carregadas de {features_path}: {self.feature_cols}")
        except FileNotFoundError:
            print(f"[WARNING] Ficheiro de features não encontrado. Usar ordem manual das features.")
            self.feature_cols = None
    
    def predict(self, company_data: Dict[str, Any]) -> int:
        """
        Prevê o cluster para uma entidade baseado nas suas métricas (1 janela).
        
        Args:
            company_data: Dicionário com as métricas da entidade.
                         Exemplo: {
                             'total_requests': 1500,
                             'mean_requests_per_minute': 25.3,
                             'unique_routes': 12,
                             'unique_api_modules': 5
                         }
        
        Returns:
            ID do cluster (int)
        """
        if self.feature_cols is None:
            raise ValueError("Features não foram carregadas. Verifique o ficheiro de features.")
        
        df = pd.DataFrame([company_data])
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Features ausentes: {missing}")
        
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        cluster_id = self.kmeans.predict(X_scaled)[0]
        return cluster_id
    
    def predict_with_confidence(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prevê o cluster e devolve a distância aos centros (confiança).
        
        Args:
            company_data: Dicionário com as métricas da entidade
        
        Returns:
            Dicionário com:
                - 'cluster': ID do cluster previsto
                - 'distances': Distâncias aos centros de cada cluster
                - 'confidence': Diferença entre 1º e 2º menor distância
        """
        if self.feature_cols is None:
            raise ValueError("Features não foram carregadas. Verifique o ficheiro de features.")
        
        df = pd.DataFrame([company_data])
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Features ausentes: {missing}")
        
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        cluster_id = self.kmeans.predict(X_scaled)[0]
        distances = self.kmeans.transform(X_scaled)[0]
        
        sorted_distances = sorted(distances)
        confidence = sorted_distances[1] - sorted_distances[0] if len(sorted_distances) > 1 else 0
        
        return {
            'cluster': cluster_id,
            'distances': distances.tolist(),
            'confidence': confidence,
            'n_clusters': self.kmeans.n_clusters
        }
    
    def predict_batch(self, batch_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prevê clusters para múltiplas janelas.
        
        Args:
            batch_data: DataFrame com múltiplas linhas (janelas de 60min)
        
        Returns:
            DataFrame com coluna 'predicted_cluster' adicionada
        """
        if self.feature_cols is None:
            raise ValueError("Features não foram carregadas. Verifique o ficheiro de features.")
        
        missing = set(self.feature_cols) - set(batch_data.columns)
        if missing:
            raise ValueError(f"Features ausentes: {missing}")
        
        X = batch_data[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        predictions = self.kmeans.predict(X_scaled)
        
        batch_data_copy = batch_data.copy()
        batch_data_copy['predicted_cluster'] = predictions
        
        return batch_data_copy
    
    def predict_summary(self, batch_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Prevê clusters para múltiplas janelas e retorna um resumo estatístico.
        
        Args:
            batch_data: DataFrame com múltiplas linhas (7 dias de 1 entidade)
        
        Returns:
            Dicionário com resumo das previsões
        """
        if self.feature_cols is None:
            raise ValueError("Features não foram carregadas. Verifique o ficheiro de features.")
        
        missing = set(self.feature_cols) - set(batch_data.columns)
        if missing:
            raise ValueError(f"Features ausentes: {missing}")
        
        X = batch_data[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        predictions = self.kmeans.predict(X_scaled)
        
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        percentages = {int(k): float(v / len(predictions) * 100) for k, v in zip(unique, counts)}
        
        dominant_cluster = int(unique[np.argmax(counts)])
        
        return {
            'n_windows': len(predictions),
            'cluster_distribution': {int(k): int(v) for k, v in distribution.items()},
            'cluster_percentages': percentages,
            'dominant_cluster': dominant_cluster,
            'n_clusters': self.kmeans.n_clusters
        }

    def predict_aggregated(self, batch_data: pd.DataFrame) -> int:
        """
        Agrega as métricas (média) tal como feito no treino do kmeans.py e devolve 1 cluster.
        Usa exatamente as mesmas features e scaler do treino → resultado mais consistente.
        """
        if self.feature_cols is None:
            raise ValueError("Features não foram carregadas. Verifique o ficheiro de features.")

        missing = set(self.feature_cols) - set(batch_data.columns)
        if missing:
            raise ValueError(f"Features ausentes: {missing}")

        aggregated_values = batch_data[self.feature_cols].mean().values.reshape(1, -1)
        aggregated_scaled = self.scaler.transform(aggregated_values)
        cluster_id = self.kmeans.predict(aggregated_scaled)[0]
        return int(cluster_id)

    def predict_aggregated_with_details(self, batch_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Igual ao predict_aggregated, mas devolve também as métricas agregadas
        e distâncias aos centróides para debug/confiança.
        """
        if self.feature_cols is None:
            raise ValueError("Features não foram carregadas. Verifique o ficheiro de features.")

        missing = set(self.feature_cols) - set(batch_data.columns)
        if missing:
            raise ValueError(f"Features ausentes: {missing}")

        aggregated_series = batch_data[self.feature_cols].mean()
        aggregated_values = aggregated_series.values.reshape(1, -1)
        aggregated_scaled = self.scaler.transform(aggregated_values)

        cluster_id = self.kmeans.predict(aggregated_scaled)[0]
        distances = self.kmeans.transform(aggregated_scaled)[0]
        sorted_distances = sorted(distances)
        confidence = sorted_distances[1] - sorted_distances[0] if len(sorted_distances) > 1 else 0

        return {
            'cluster': int(cluster_id),
            'aggregated_metrics': dict(zip(self.feature_cols, aggregated_series.tolist())),
            'distances': distances.tolist(),
            'confidence': confidence,
            'n_windows_aggregated': len(batch_data),
            'n_clusters': self.kmeans.n_clusters,
        }

    # -----------------
    # Plot helpers
    # -----------------
    def plot_cluster_timeline(self, df_with_preds: pd.DataFrame, company_id: str):
        """Plota a série temporal de clusters por janela (usa coluna @timestamp se existir)."""
        if '@timestamp' not in df_with_preds.columns:
            print('[PLOT] Coluna @timestamp não disponível; a linha temporal não será plotada.')
            return

        df_plot = df_with_preds.copy()
        df_plot['@timestamp'] = pd.to_datetime(df_plot['@timestamp'])
        df_plot = df_plot.sort_values('@timestamp')

        plt.figure(figsize=(10, 4))
        plt.plot(df_plot['@timestamp'], df_plot['predicted_cluster'], marker='o', linestyle='-')
        plt.yticks(range(self.kmeans.n_clusters))
        plt.title(f'Clusters por janela - {company_id}')
        plt.xlabel('Tempo')
        plt.ylabel('Cluster previsto')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictor_timeline_clusters.png', bbox_inches='tight')
        print('[PLOT] Gráfico de timeline guardado em: timeline_clusters.png')
        plt.close()

    def plot_aggregated_distances(self, agg_result: Dict[str, Any]):
        """Plota barras com distâncias aos centróides para a previsão agregada."""
        distances = agg_result.get('distances')
        if not distances:
            print('[PLOT] Distâncias não disponíveis para plot.')
            return

        labels = [f'C{i}' for i in range(len(distances))]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, distances, color='steelblue')
        plt.title('Distância aos centróides (ponto agregado)')
        plt.ylabel('Distância (espaço escalado)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictor_aggregated_distances.png', bbox_inches='tight')
        print('[PLOT] Gráfico de distâncias guardado em: aggregated_distances.png')
        plt.close()
    
    @staticmethod
    def parse_iso_utc(value: str) -> datetime:
        """Parse ISO-8601 string forcing timezone UTC."""
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    
    @staticmethod
    def get_es_client(hosts: Optional[Sequence[str]] = None,
                     user: Optional[str] = None,
                     password: Optional[str] = None) -> OpenSearch:
        """Cria e devolve um cliente OpenSearch autenticado."""
        hosts = list(hosts) if hosts is not None else ES_HOSTS
        user = user if user is not None else ES_USER
        password = password if password is not None else ES_PASSWORD

        client = OpenSearch(
            hosts=hosts,
            http_auth=(user, password),
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=60,
        )
        if not client.ping():
            raise ConnectionError("Falha na autenticação no Wazuh Indexer.")
        return client
    
    def fetch_company_metrics(self, company_id: str, start_dt: datetime, end_dt: datetime,
                             client: Optional[OpenSearch] = None) -> pd.DataFrame:
        """
        Vai buscar todas as métricas L1 de uma entidade num intervalo de tempo.
        
        Args:
            company_id: ID da entidade (ex: 'ENTITY_001')
            start_dt: Data/hora início (datetime UTC)
            end_dt: Data/hora fim (datetime UTC)
            client: Cliente OpenSearch (se None, cria um novo)
        
        Returns:
            DataFrame com as métricas
        """
        if client is None:
            client = self.get_es_client()
        
        print(f"\n[FETCH] A buscar métricas para a entidade {company_id} entre {start_dt} e {end_dt}...")
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_dt.isoformat(),
                                    "lt": end_dt.isoformat(),
                                }
                            }
                        },
                        {
                            "term": {
                                "company_id.keyword": company_id
                            }
                        }
                    ]
                }
            }
        }
        
        # Scroll para ir buscar tudo
        response = client.search(
            index=METRICS_INDEX_PATTERN,
            body=query,
            scroll="2m",
            size=1000,
        )
        
        hits = []
        scroll_id = response.get("_scroll_id")
        
        while True:
            batch = response.get("hits", {}).get("hits", [])
            if not batch:
                break
            hits.extend(batch)
            
            response = client.scroll(scroll_id=scroll_id, body={"scroll": "2m"})
            scroll_id = response.get("_scroll_id")
        
        print(f"[FETCH] Encontradas {len(hits)} janelas para a entidade {company_id}")
        
        if len(hits) == 0:
            print(f"[WARNING] Nenhuma métrica encontrada para {company_id} neste intervalo.")
            return pd.DataFrame()
        
        # Flatten os dados - extrair apenas as features necessárias
        rows = []
        for doc in hits:
            source = doc.get("_source", {})
            row = {
                "company_id": source.get("company_id"),
                "@timestamp": source.get("@timestamp") or source.get("timestamp"),
            }
            
            # Extrair apenas as métricas necessárias (as 4 features do modelo)
            metrics = source.get("metrics", {})
            if self.feature_cols:
                for feature in self.feature_cols:
                    row[feature] = metrics.get(feature, 0)
            else:
                # Fallback: extrair tudo se não tiver feature_cols
                for key, val in metrics.items():
                    row[key] = val
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"[OK] DataFrame criado com {len(df)} linhas e {len(df.columns)} colunas")
        print(f"[OK] Colunas: {df.columns.tolist()}")
        
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Faz previsões de cluster para uma entidade num intervalo de 7 dias."
    )
    parser.add_argument(
        "--company-id",
        type=str,
        default="125",
        help="ID da entidade (ex: 125)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-10-01T00:00:00",
        help="Data/hora início (ISO, UTC) - ex: 2025-09-01T00:00:00",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-10-08T00:00:00",
        help="Data/hora fim (ISO, UTC) - ex: 2025-09-08T00:00:00",
    )
    parser.add_argument(
        "--es-hosts",
        type=str,
        default=None,
        help="Hosts do OpenSearch separados por vírgulas",
    )
    parser.add_argument(
        "--es-user",
        type=str,
        default=None,
        help="User do OpenSearch",
    )
    parser.add_argument(
        "--es-password",
        type=str,
        default=None,
        help="Password do OpenSearch",
    )
    
    args = parser.parse_args()
    
    try:
        # Inicializar preditor
        predictor = KMeansPredictor()
        
        # Parse datas
        start_dt = predictor.parse_iso_utc(args.start)
        end_dt = predictor.parse_iso_utc(args.end)
        
        # Conectar ao Wazuh
        client = predictor.get_es_client(
            hosts=args.es_hosts.split(",") if args.es_hosts else None,
            user=args.es_user,
            password=args.es_password
        )
        
        # Fetch dos dados
        df_metrics = predictor.fetch_company_metrics(
            args.company_id,
            start_dt,
            end_dt,
            client
        )
        
        if df_metrics.empty:
            print("[ERROR] Nenhuma métrica encontrada. Abort.")
            exit(1)
        
        # Fazer previsões
        print(f"\n[PREDICT] A fazer previsões para {len(df_metrics)} janelas...")
        df_com_predicoes = predictor.predict_batch(df_metrics)
        
        # Mostrar algumas previsões (com todas as features)
        print(f"\n[RESULTADO] Primeiras 10 previsões:")
        feature_cols = ['total_requests', 'mean_requests_per_minute', 'unique_routes', 'unique_api_modules', 'predicted_cluster']
        available_cols = [c for c in feature_cols if c in df_com_predicoes.columns]
        print(df_com_predicoes[available_cols].head(10))
        
        # Resumo
        summary = predictor.predict_summary(df_metrics)
        print(f"\n[RESUMO] Análise de 7 dias para {args.company_id}:")
        print(f"  Total de janelas: {summary['n_windows']}")
        print(f"  Cluster dominante: {summary['dominant_cluster']}")
        print(f"  Distribuição: {summary['cluster_distribution']}")
        print(f"  Percentagens: {summary['cluster_percentages']}")

        # Previsão agregada (consistente com o treino: média das 168 janelas)
        agg_result = predictor.predict_aggregated_with_details(df_metrics)
        print(f"\n[AGG] Previsão agregada (média das janelas): {agg_result['cluster']}")
        print(f"[AGG] Métricas agregadas: {agg_result['aggregated_metrics']}")
        print(f"[AGG] Distâncias aos centros: {[f'{d:.4f}' for d in agg_result['distances']]}")
        print(f"[AGG] Confiança (diferença distâncias): {agg_result['confidence']:.4f}")

        # Gráficos
        predictor.plot_cluster_timeline(df_com_predicoes, args.company_id)
        predictor.plot_aggregated_distances(agg_result)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
