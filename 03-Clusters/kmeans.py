import argparse
import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from opensearchpy import OpenSearch
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from joblib import dump

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", "Unverified HTTPS request")

# --- CONFIG WAZUH / OPENSEARCH ---
ES_HOSTS = os.getenv("ES_HOSTS", "https://100.125.228.80:9200").split(",")
ES_USER = os.getenv("ES_USER", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "SecretPassword")

METRICS_INDEX_PATTERN = "metrics-l1-60m*"
DEFAULT_FEATURES = [
    "total_requests",
    "mean_requests_per_minute",
    #"unique_source_ips",
    "unique_routes",
    "unique_api_modules",
    #"mean_requests_per_ip",
    #"mean_response_time",
]


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Devolve todas as colunas de feature (exclui `company_id`)."""
    return [c for c in df.columns if c != "company_id"]


def parse_iso_utc(value: str) -> datetime:
    """Parse ISO-8601 string forcing timezone UTC."""
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


# ==========================
# 1. Conexão ao OpenSearch
# ==========================
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


# ==========================
# 2. Fetch de métricas L1
# ==========================
def fetch_metrics(client: OpenSearch, start_dt: datetime, end_dt: datetime,
                  index_pattern: str = METRICS_INDEX_PATTERN) -> pd.DataFrame:
    """
    Vai buscar TODOS os documentos de métricas L1 entre start_dt e end_dt.

    Retorna um DataFrame com as colunas flatten:
    - @timestamp
    - company_id
    - metrics.total_requests
    - ...
    """
    print(f"\n[FETCH] A buscar métricas em {index_pattern} entre {start_dt} e {end_dt}...")

    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": start_dt.isoformat(),
                    "lt": end_dt.isoformat(),
                }
            }
        }
    }

    # Scroll para ir buscar tudo
    response = client.search(
        index=index_pattern,
        body=query,
        size=1000,
        scroll="2m",
    )

    scroll_id = response.get("_scroll_id")
    hits = response["hits"]["hits"]

    all_hits = hits.copy()

    while scroll_id and len(hits) > 0:
        response = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]
        all_hits.extend(hits)

    if not all_hits:
        print("[FETCH] Não foram encontrados documentos neste intervalo.")
        return pd.DataFrame()

    sources = [h["_source"] for h in all_hits]
    df = pd.json_normalize(sources, sep=".")
    print(f"[FETCH] Foram carregados {len(df)} documentos brutos.")

    # Garantir que @timestamp existe e está em datetime
    if "@timestamp" in df.columns:
        df["@timestamp"] = pd.to_datetime(df["@timestamp"])
    else:
        # fallback se o campo vier só como 'timestamp'
        if "timestamp" in df.columns:
            df["@timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            raise KeyError("Nenhum campo '@timestamp' ou 'timestamp' encontrado nos documentos.")

    # Remover duplicados por company_id + @timestamp, caso existam reprocessamentos
    if "company_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["company_id", "@timestamp"])
        after = len(df)
        if after < before:
            print(f"[CLEAN] Removidos {before - after} duplicados (company_id, @timestamp).")
    else:
        raise KeyError("Campo 'company_id' não encontrado nas métricas.")

    return df


# ==========================
# 3. Construção de features
# ==========================
def build_company_feature_table(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """
    A partir de um DF de docs de métricas, constrói uma tabela
    com UMA LINHA por company_id, agregando as métricas por média.
    """
    if df.empty:
        print("[BUILD] DataFrame vazio, nada para agregar.")
        return pd.DataFrame()

    # Garantir que temos as colunas metrics.X
    missing_cols = [f"metrics.{f}" for f in feature_names if f"metrics.{f}" not in df.columns]
    if missing_cols:
        print("[WARN] Faltam as seguintes colunas de métricas no índice:")
        for c in missing_cols:
            print("   -", c)
        # Vamos continuar, mas só com as que existem
    cols_present = [f"metrics.{f}" for f in feature_names if f"metrics.{f}" in df.columns]

    if not cols_present:
        print("[BUILD] Nenhuma das features pretendidas está disponível. Abort.")
        return pd.DataFrame()

    # Subconjunto com as colunas relevantes
    sub = df[["company_id"] + cols_present].copy()

    # Converter para numérico e substituir NaN por 0 (ou podes optar por dropna)
    for c in cols_present:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub[cols_present] = sub[cols_present].fillna(0)

    # Agregar por empresa: média ao longo do período
    grouped = sub.groupby("company_id")[cols_present].mean().reset_index()

    # Renomear as colunas para o nome simples da feature (sem "metrics.")
    rename_map = {f"metrics.{f}": f for f in feature_names if f"metrics.{f}" in grouped.columns}
    grouped = grouped.rename(columns=rename_map)

    print(f"[BUILD] Construída tabela de features com {len(grouped)} empresas.")
    return grouped


# ==========================
# 4. Treino do K-Means
# ==========================
def train_kmeans(df_features: pd.DataFrame, n_clusters: int):
    """
    Treina um K-Means em df_features (sem a coluna company_id).
    Retorna (scaler, kmeans, df_resultados, silhouette_train).
    """
    feature_cols = feature_columns(df_features)
    X = df_features[feature_cols].values

    # Normalizar de forma robusta (menos sensível a outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=25,
    )
    labels = kmeans.fit_predict(X_scaled)

    silhouette_train = compute_silhouette(X_scaled, labels)

    df_out = df_features.copy()
    df_out["cluster"] = labels

    return scaler, kmeans, df_out, silhouette_train


# ==========================
# 5. Aplicar K-Means a outro período
# ==========================
def apply_kmeans(df_features: pd.DataFrame, scaler: RobustScaler, kmeans: KMeans) -> pd.DataFrame:
    """Aplica o scaler + modelo a um novo DF de features por empresa."""
    feature_cols = feature_columns(df_features)
    X = df_features[feature_cols].values
    X_scaled = scaler.transform(X)
    labels = kmeans.predict(X_scaled)

    df_out = df_features.copy()
    df_out["cluster"] = labels
    return df_out


# ==========================
# 6. Métrica de estabilidade
# ==========================
def compute_cluster_stability(train_clusters: pd.DataFrame, test_clusters: pd.DataFrame):
    """
    Calcula a percentagem de empresas que mantêm o mesmo cluster
    entre treino e teste (apenas para empresas que existem nos dois).
    """
    merged = pd.merge(
        train_clusters[["company_id", "cluster"]],
        test_clusters[["company_id", "cluster"]],
        on="company_id",
        suffixes=("_train", "_test"),
    )

    if merged.empty:
        print("[STABILITY] Não há empresas em comum entre treino e teste.")
        return None

    stability = (merged["cluster_train"] == merged["cluster_test"]).mean()
    print(f"[STABILITY] Empresas em comum: {len(merged)}")
    print(f"[STABILITY] Percentagem que manteve o cluster: {stability * 100:.2f}%")

    return stability


def cluster_distance_diagnostics(
    df_features: pd.DataFrame,
    df_clusters: pd.DataFrame,
    scaler: RobustScaler,
    kmeans: KMeans,
    top_n: int = 20,
    label: str = "Treino",
    sub_target_cluster: Optional[int] = None,
):
    """
    Calcula, para cada empresa:
      - distância ao centróide do cluster atribuído
      - distância ao centróide alternativo mais próximo
      - margin = dist_outro - dist_atribuído  (quanto maior, mais confiança)
      - ratio  = dist_atribuído / dist_outro (quanto menor, melhor)

    Imprime as empresas mais ambíguas (menor margin) e devolve um DataFrame
    com os resultados para análise em Jupyter.
    """
    # Garantir alinhamento (company_id, cluster)
    merged = pd.merge(
        df_features,
        df_clusters[["company_id", "cluster"]],
        on="company_id",
        how="inner",
    )

    if merged.empty:
        print(f"[DIST] {label}: nenhum dado para diagnóstico.")
        return pd.DataFrame()

    feature_cols = feature_columns(df_features)

    X = merged[feature_cols].values
    X_scaled = scaler.transform(X)
    centers = kmeans.cluster_centers_  # no espaço escalado

    labels = merged["cluster"].values
    company_ids = merged["company_id"].values

    # Distância de cada ponto a todos os centróides: shape (n_samples, k)
    # d(i, j) = ||x_i - center_j||
    diff = X_scaled[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Distância ao centróide atribuído
    idx = np.arange(len(labels))
    dist_assigned = dists[idx, labels]

    # Distância ao centróide alternativo mais próximo (2º mais perto)
    # argsort dá índices ordenados por distância
    nearest_order = np.argsort(dists, axis=1)
    nearest_other_cluster = nearest_order[:, 1]
    dist_nearest_other = dists[idx, nearest_other_cluster]

    margin = dist_nearest_other - dist_assigned   # quanto maior, melhor
    ratio = dist_assigned / (dist_nearest_other + 1e-9)

    diag_df = pd.DataFrame({
        "company_id": company_ids,
        "cluster": labels,
        "dist_assigned": dist_assigned,
        "nearest_other_cluster": nearest_other_cluster,
        "dist_nearest_other": dist_nearest_other,
        "margin": margin,
        "ratio": ratio,
    })

    diag_df_sorted = diag_df.sort_values("margin")

    print(f"\n[DIST] Diagnóstico de distâncias ({label})")
    print(f"[DIST] Empresas analisadas: {len(diag_df)}")
    print(f"[DIST] Margin média: {diag_df['margin'].mean():.4f}")
    print(f"[DIST] Top {top_n} empresas mais ambíguas (menor margin):")
    print(diag_df_sorted.head(top_n).to_string(index=False))

    return diag_df


# ==========================
# 6.b Subcluster dentro de um cluster existente
# ==========================
def train_subcluster_inside_cluster(
    df_features: pd.DataFrame,
    df_clusters: pd.DataFrame,
    scaler: RobustScaler,
    target_cluster: int = 0,
    sub_k: int = 2,
):
    """
    Treina um K-Means apenas nos pontos do cluster `target_cluster`,
    mantendo o clustering principal inalterado. Retorna (kmeans_sub, df_clusters_out).
    """
    merged = pd.merge(
        df_features,
        df_clusters[["company_id", "cluster"]],
        on="company_id",
        how="inner",
    )

    subset = merged[merged["cluster"] == target_cluster]
    if subset.empty:
        print(f"[SUB] Cluster {target_cluster} está vazio. Não será subdividido.")
        return None, df_clusters

    if len(subset) < sub_k:
        print(f"[SUB] Cluster {target_cluster} tem menos empresas ({len(subset)}) do que sub_k={sub_k}. Abort subcluster.")
        return None, df_clusters

    feature_cols = [c for c in subset.columns if c not in ("company_id", "cluster")]
    X = subset[feature_cols].values
    X_scaled = scaler.transform(X)

    kmeans_sub = KMeans(n_clusters=sub_k, random_state=42, n_init=25)
    sub_labels = kmeans_sub.fit_predict(X_scaled)

    # Mapear de volta por company_id para evitar desalinhamentos de ordem
    sub_df = pd.DataFrame({"company_id": subset["company_id"].values, "sub_label": sub_labels})

    df_out = df_clusters.copy()
    sub_col = f"sub_{target_cluster}"
    df_out[sub_col] = np.nan
    df_out = df_out.merge(sub_df, on="company_id", how="left")
    df_out[sub_col] = df_out[sub_col].fillna(df_out.pop("sub_label"))

    print(f"[SUB] Subdivisão do cluster {target_cluster} em {sub_k} subclusters concluída.")
    print(df_out.loc[df_out["cluster"] == target_cluster, sub_col].value_counts().sort_index())

    return kmeans_sub, df_out


def apply_subcluster(
    df_features: pd.DataFrame,
    df_clusters: pd.DataFrame,
    scaler: RobustScaler,
    kmeans_sub: Optional[KMeans],
    target_cluster: int = 0,
):
    """
    Aplica um modelo de subcluster apenas aos pontos que pertencem ao cluster `target_cluster`.
    """
    if kmeans_sub is None:
        return df_clusters

    # Selecionar apenas empresas do cluster alvo
    target_ids = df_clusters.loc[df_clusters["cluster"] == target_cluster, "company_id"]
    if target_ids.empty:
        print(f"[SUB] Nenhuma empresa com cluster {target_cluster} para subclusterizar no novo período.")
        df_out = df_clusters.copy()
        df_out[f"sub_{target_cluster}"] = np.nan
        return df_out

    merged = pd.merge(
        df_features,
        target_ids.to_frame(),
        on="company_id",
        how="inner",
    )

    feature_cols = feature_columns(merged)
    X = merged[feature_cols].values
    X_scaled = scaler.transform(X)

    sub_labels = kmeans_sub.predict(X_scaled)
    sub_df = pd.DataFrame({"company_id": merged["company_id"].values, "sub_label": sub_labels})

    df_out = df_clusters.copy()
    sub_col = f"sub_{target_cluster}"
    df_out[sub_col] = np.nan
    df_out = df_out.merge(sub_df, on="company_id", how="left")
    df_out[sub_col] = df_out[sub_col].fillna(df_out.pop("sub_label"))

    return df_out



# ==========================
# 7. Gráfico PCA dos clusters (visualização)
# ==========================
def plot_clusters_pca(df_features, df_clusters, scaler, kmeans, title, filename=None, sub_target_cluster=None):
    """
    Desenha um gráfico 2D (PCA) dos clusters, com:
      - Pontos coloridos por cluster
      - Centróides do K-Means destacados com 'X'

    PCA é usado apenas para visualização (não faz parte do treino).
    """
    # Garantir que os dados estão alinhados por empresa
    merge_cols = ["company_id", "cluster"]
    if sub_target_cluster is not None:
        sub_col = f"sub_{sub_target_cluster}"
        if sub_col in df_clusters.columns:
            merge_cols.append(sub_col)

    merged = pd.merge(
        df_features,
        df_clusters[merge_cols],
        on="company_id",
        how="inner",
    )

    if merged.empty:
        print("[PLOT] Não há empresas para plotar.")
        return

    feature_cols = [c for c in df_features.columns if c != "company_id"]
    expected = getattr(scaler, "n_features_in_", len(feature_cols))
    if len(feature_cols) != expected:
        print(f"[PLOT] Aviso: {len(feature_cols)} colunas de features, mas scaler espera {expected}. Ajustando...")
        feature_cols = feature_cols[:expected]

    X = merged[feature_cols].values
    X_scaled = scaler.transform(X)

    # PCA 2D só para visualização
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    clusters = merged["cluster"].values

    plt.figure(figsize=(10, 8))
    
    # Criar background com regiões de cada cluster
    # Criar uma malha de pontos para classificar
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    
    # Resolução da malha (ajustar se necessário para mais/menos detalhe)
    h = 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    # Projetar a malha de volta ao espaço original para classificação
    # Precisamos fazer o inverso da transformação PCA
    mesh_points_pca = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = pca.inverse_transform(mesh_points_pca)
    
    # Classificar cada ponto da malha usando o modelo K-Means
    Z = kmeans.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Desenhar o contorno preenchido (background)
    plt.contourf(xx, yy, Z, levels=np.arange(-0.5, kmeans.n_clusters + 0.5, 1),
                 alpha=0.3, cmap='viridis')
    
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters,
        alpha=0.75,
        s=40,
        edgecolors='black',
        linewidth=0.5,
    )

    # Se existir subclusterização para o cluster alvo, sobrepor com outra paleta
    if sub_target_cluster is not None:
        sub_col = f"sub_{sub_target_cluster}"
        if sub_col in merged.columns:
            mask_sub = (merged["cluster"] == sub_target_cluster) & merged[sub_col].notna()
            if mask_sub.any():
                subs = merged.loc[mask_sub, sub_col].values
                plt.scatter(
                    X_pca[mask_sub, 0],
                    X_pca[mask_sub, 1],
                    c=subs,
                    cmap='tab10',
                    alpha=0.9,
                    s=55,
                    marker='o',
                    edgecolors='white',
                    linewidth=0.6,
                    label=f"Subclusters do cluster {sub_target_cluster}",
                )

    # Centróides: no espaço escalado -> projetar para PCA
    centers_scaled = kmeans.cluster_centers_
    centers_pca = pca.transform(centers_scaled)

    plt.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        marker="X",
        s=250,
        edgecolor="black",
        linewidths=2,
        facecolor="white",
        label="Centróide",
    )

    # Legendas
    legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.gca().add_artist(legend1)
    plt.legend(loc="best")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        print(f"[PLOT] Gráfico guardado em: {filename}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# ==========================
# 7.b Gráfico PCA para subcluster de um cluster específico
# ==========================
def plot_subcluster_pca(df_features, df_clusters, scaler, kmeans_sub, target_cluster, title, filename=None):
    """
    Desenha um gráfico PCA apenas para o cluster `target_cluster`, colorindo pelos subclusters.
    """
    sub_col = f"sub_{target_cluster}"
    if sub_col not in df_clusters.columns:
        print(f"[PLOT] Coluna {sub_col} não existe, não há subclusters para plotar.")
        return

    merged = pd.merge(
        df_features,
        df_clusters[["company_id", "cluster", sub_col]],
        on="company_id",
        how="inner",
    )

    subset = merged[(merged["cluster"] == target_cluster) & (merged[sub_col].notna())]
    if subset.empty:
        print(f"[PLOT] Nenhuma empresa no cluster {target_cluster} com subcluster definido.")
        return

    feature_cols = [c for c in subset.columns if c not in ("company_id", "cluster", sub_col)]

    X = subset[feature_cols].values
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    subs = subset[sub_col].values

    plt.figure(figsize=(8, 6))

    # Malha de decisão para os subclusters
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    mesh_points_pca = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = pca.inverse_transform(mesh_points_pca)
    Z = kmeans_sub.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=np.arange(-0.5, kmeans_sub.n_clusters + 0.5, 1),
                 alpha=0.25, cmap='tab10')

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=subs,
        cmap='tab10',
        alpha=0.8,
        s=50,
        edgecolors='black',
        linewidth=0.6,
    )

    centers_scaled = kmeans_sub.cluster_centers_
    centers_pca = pca.transform(centers_scaled)
    plt.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        marker="X",
        s=200,
        edgecolor="black",
        linewidths=2,
        facecolor="white",
        label="Centro subcluster",
    )

    legend1 = plt.legend(*scatter.legend_elements(), title=f"Subcluster {target_cluster}")
    plt.gca().add_artist(legend1)
    plt.legend(loc="best")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        print(f"[PLOT] Gráfico de subcluster guardado em: {filename}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# ==========================
# 8. main()
# ==========================
def compute_silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Silhouette seguro: devolve None se não for calculável."""
    if len(np.unique(labels)) <= 1:
        return None
    return silhouette_score(X_scaled, labels)


def main(
    train_start_str: str,
    train_end_str: str,
    test_start_str: str,
    test_end_str: str,
    n_clusters: int,
    sub_target_cluster: int = 0,
    sub_k: int = 2,
    feature_names: Optional[List[str]] = None,
    es_hosts: Optional[Sequence[str]] = None,
    es_user: Optional[str] = None,
    es_password: Optional[str] = None,
):
    # 1) Converter strings para datetimes com timezone UTC
    train_start = parse_iso_utc(train_start_str)
    train_end = parse_iso_utc(train_end_str)
    test_start = parse_iso_utc(test_start_str)
    test_end = parse_iso_utc(test_end_str)

    # 2) Features escolhidas (aqui está a versão que te deu score alto)
    feature_names = feature_names or DEFAULT_FEATURES

    # 3) Conectar ao ES
    print("[INFO] A conectar ao Wazuh Indexer...")
    client = get_es_client(hosts=es_hosts, user=es_user, password=es_password)
    print("[INFO] Conectado com sucesso.")

    # 4) Fetch + build treino
    df_train_raw = fetch_metrics(client, train_start, train_end)
    df_train_feat = build_company_feature_table(df_train_raw, feature_names)

    if df_train_feat.empty:
        print("[MAIN] Não há dados de treino. Abort.")
        return

    # 5) Treinar K-Means
    print(f"\n[TRAIN] A treinar K-Means com k={n_clusters} clusters...")
    scaler, kmeans, df_train_clusters, silhouette_train = train_kmeans(df_train_feat, n_clusters)
    print("[TRAIN] Treino concluído.")
    print("[TRAIN] Distribuição de empresas por cluster (treino):")
    print(df_train_clusters["cluster"].value_counts().sort_index())

    if silhouette_train is not None:
        print(f"[TRAIN] Silhouette score (treino): {silhouette_train:.4f}")
    else:
        print("[TRAIN] Silhouette score (treino) não pôde ser calculado (clusters insuficientes ou degenerados).")

    # 5.b) Subcluster dentro de um cluster específico (opcional)
    kmeans_sub = None
    if sub_k and sub_k > 1 and n_clusters >= 1:
        kmeans_sub, df_train_clusters = train_subcluster_inside_cluster(
            df_train_feat,
            df_train_clusters,
            scaler,
            target_cluster=sub_target_cluster,
            sub_k=sub_k,
        )
    else:
        print(f"[SUB] Subcluster desativado (sub_k={sub_k}).")

    # 6) Fetch + build teste
    df_test_raw = fetch_metrics(client, test_start, test_end)
    df_test_feat = build_company_feature_table(df_test_raw, feature_names)

    if df_test_feat.empty:
        print("[MAIN] Não há dados de teste. Só clusters de treino foram gerados.")
        return

    # 7) Aplicar modelo ao período de teste
    print("\n[TEST] A aplicar K-Means aos dados de teste...")
    df_test_clusters = apply_kmeans(df_test_feat, scaler, kmeans)

    # Aplicar subcluster ao cluster 0 no período de teste
    df_test_clusters = apply_subcluster(
        df_test_feat,
        df_test_clusters,
        scaler,
        kmeans_sub,
        target_cluster=sub_target_cluster,
    )

    print("[TEST] Distribuição de empresas por cluster (teste):")
    print(df_test_clusters["cluster"].value_counts().sort_index())

    if kmeans_sub is not None:
        sub_col = f"sub_{sub_target_cluster}"
        print(f"[TEST] Subclusters dentro do cluster {sub_target_cluster} (teste):")
        print(df_test_clusters.loc[df_test_clusters["cluster"] == sub_target_cluster, sub_col]
              .value_counts().sort_index())

    # 8) Silhouette no conjunto de teste
    feature_cols = feature_columns(df_test_feat)
    X_test = df_test_feat[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    labels_test = df_test_clusters["cluster"].values

    silhouette_test = compute_silhouette(X_test_scaled, labels_test)

    if silhouette_test is not None:
        print(f"[TEST] Silhouette score (teste): {silhouette_test:.4f}")
    else:
        print("[TEST] Silhouette score (teste) não pôde ser calculado (clusters insuficientes ou degenerados).")

    # 9) Estabilidade entre períodos
    print("\n[ANALYSIS] Estabilidade dos clusters entre treino e teste:")
    compute_cluster_stability(df_train_clusters, df_test_clusters)

    # 10) Centros dos clusters (espaço original, para interpretação)
    feature_cols_train = feature_columns(df_train_feat)

    centers_scaled = kmeans.cluster_centers_             # (k, n_features escaladas)
    centers = scaler.inverse_transform(centers_scaled)   # voltar ao espaço original

    centers_df = pd.DataFrame(centers, columns=feature_cols_train)
    centers_df["cluster"] = centers_df.index

    print("\n[CENTERS] Centros dos clusters (em escala original):")
    print(centers_df.set_index("cluster").round(2))

    # 11) Gráficos PCA (apenas visualização)
    print("\n[PLOT] A gerar gráfico 2D (PCA) para os clusters de treino...")
    plot_clusters_pca(
        df_train_feat,
        df_train_clusters,
        scaler,
        kmeans,
        title="Clusters K-Means (Treino)",
        filename="kmeans_clusters_treino.png",
        sub_target_cluster=sub_target_cluster,
    )

    print("\n[PLOT] A gerar gráfico 2D (PCA) para os clusters de teste...")
    plot_clusters_pca(
        df_test_feat,
        df_test_clusters,
        scaler,
        kmeans,
        title="Clusters K-Means (Teste)",
        filename="kmeans_clusters_teste.png",
        sub_target_cluster=sub_target_cluster,
    )

    # 12) Guardar modelo e scaler para previsão futura
    print("\n[MODEL] A guardar modelo e scaler...")
    dump(kmeans, 'kmeans_model.joblib')
    dump(scaler, 'scaler_model.joblib')
    
    # Guardar também os nomes das features para referência
    feature_cols_train = feature_columns(df_train_feat)
    with open('kmeans_features.txt', 'w') as f:
        f.write(','.join(feature_cols_train))
    
    print(f"[MODEL] Modelo guardado em: kmeans_model.joblib")
    print(f"[MODEL] Scaler guardado em: scaler_model.joblib")
    print(f"[MODEL] Features guardadas em: kmeans_features.txt")

    # Para uso em Jupyter, se quiseres importar main()
    return df_train_feat, df_train_clusters, df_test_feat, df_test_clusters, scaler, kmeans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino e teste de K-Means nas métricas L1 do Wazuh.")
    parser.add_argument(
        "--train-start",
        type=str,
        default="2025-10-01T00:00:00",
        help="Início do período de treino (ISO, UTC).",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2025-10-16T00:00:00",
        help="Fim do período de treino (exclusivo, ISO, UTC).",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="2025-10-16T00:00:00",
        help="Início do período de teste (ISO, UTC).",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default="2025-11-01T00:00:00",
        help="Fim do período de teste (exclusivo, ISO, UTC).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Número de clusters do K-Means (ex: 3 para pequeno/médio/grande).",
    )

    parser.add_argument(
        "--sub-target-cluster",
        type=int,
        default=0,
        help="Cluster principal que será subdividido (hierárquico).",
    )

    parser.add_argument(
        "--sub-k",
        type=int,
        default=0,
        help="Número de subclusters dentro do cluster alvo (se <=1, desativa subcluster).",
    )

    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Lista de features separadas por vírgulas; se omitido usa default.",
    )

    parser.add_argument(
        "--es-hosts",
        type=str,
        default=None,
        help="Hosts do OpenSearch separados por vírgulas (sobrepõe env ES_HOSTS).",
    )

    parser.add_argument(
        "--es-user",
        type=str,
        default=None,
        help="User do OpenSearch (sobrepõe env ES_USER).",
    )

    parser.add_argument(
        "--es-password",
        type=str,
        default=None,
        help="Password do OpenSearch (sobrepõe env ES_PASSWORD).",
    )

    args = parser.parse_args()

    main(
        train_start_str=args.train_start,
        train_end_str=args.train_end,
        test_start_str=args.test_start,
        test_end_str=args.test_end,
        n_clusters=args.n_clusters,
        sub_target_cluster=args.sub_target_cluster,
        sub_k=args.sub_k,
        feature_names=[f.strip() for f in args.features.split(",")] if args.features else None,
        es_hosts=args.es_hosts.split(",") if args.es_hosts else None,
        es_user=args.es_user,
        es_password=args.es_password,
    )
