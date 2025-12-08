import argparse
import pandas as pd
from opensearchpy import OpenSearch
from tqdm import tqdm
from config import OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASSWORD


def get_es_client():
    client = OpenSearch(
        hosts=[OPENSEARCH_HOST],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=60,
    )
    if not client.ping():
        raise ConnectionError("Failed to connect to OpenSearch")
    return client

def analyze_entity_coverage(window, start_date, end_date, min_samples=2000):
    print(f"Analyzing entity data coverage...")
    print(f"  Window: {window}m")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Minimum samples threshold: {min_samples}")

    client = get_es_client()
    index_pattern = f"metrics-l1-{window}m*"

    query = {
        "size": 0,
        "query": {
            "range": {
                "@timestamp": {
                    "gte": start_date,
                    "lt": end_date
                }
            }
        },
        "aggs": {
            "entities": {
                "terms": {
                    "field": "company_id",
                    "size": 10000,
                    "order": {"_count": "desc"}
                }
            }
        }
    }

    print(f"\nQuerying {index_pattern}...")
    response = client.search(index=index_pattern, body=query)

    buckets = response['aggregations']['entities']['buckets']

    data = []
    for bucket in buckets:
        data.append({
            'company_id': bucket['key'],
            'sample_count': bucket['doc_count']
        })

    df = pd.DataFrame(data)

    df['sufficient_data'] = df['sample_count'] >= min_samples
    df['model_type'] = df['sufficient_data'].apply(
        lambda x: 'entity_model' if x else 'cluster_model'
    )

    total_entities = len(df)
    sufficient_count = df['sufficient_data'].sum()
    insufficient_count = total_entities - sufficient_count

    print("\n" + "="*60)
    print("ENTITY DATA ANALYSIS")
    print("="*60)
    print(f"Total entities: {total_entities}")
    print(f"  Sufficient data (>= {min_samples} samples): {sufficient_count} ({sufficient_count/total_entities*100:.1f}%)")
    print(f"  Insufficient data: {insufficient_count} ({insufficient_count/total_entities*100:.1f}%)")
    print()
    print(f"Sample distribution:")
    print(f"  Min: {df['sample_count'].min()}")
    print(f"  Max: {df['sample_count'].max()}")
    print(f"  Mean: {df['sample_count'].mean():.0f}")
    print(f"  Median: {df['sample_count'].median():.0f}")
    print()
    print(f"Percentiles:")
    print(f"  p10: {df['sample_count'].quantile(0.10):.0f}")
    print(f"  p25: {df['sample_count'].quantile(0.25):.0f}")
    print(f"  p50: {df['sample_count'].quantile(0.50):.0f}")
    print(f"  p75: {df['sample_count'].quantile(0.75):.0f}")
    print(f"  p90: {df['sample_count'].quantile(0.90):.0f}")
    print("="*60)

    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze entity data availability')
    parser.add_argument('--window', type=int, required=True, choices=[10, 30, 60],
                        help='Time window in minutes')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (ISO 8601)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (ISO 8601)')
    parser.add_argument('--min-samples', type=int, default=2000,
                        help='Minimum samples for individual model (default: 2000)')
    parser.add_argument('--output', type=str, default='entity_data_analysis.csv',
                        help='Output CSV file (default: entity_data_analysis.csv)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Show top N entities (default: 20)')

    args = parser.parse_args()

    df = analyze_entity_coverage(args.window, args.start, args.end, args.min_samples)

    df.to_csv(args.output, index=False)
    print(f"\n[OK] Full results saved to: {args.output}")

    print(f"\nTop {args.top_n} entities by sample count:")
    print(df.head(args.top_n).to_string(index=False))

    eligible = df[df['sufficient_data']].copy()
    if len(eligible) > 0:
        print(f"\n\nEntities eligible for INDIVIDUAL models ({len(eligible)} total):")
        print("Copy these IDs to train entity-specific models:")
        print("-" * 60)
        for i, row in enumerate(eligible.head(20).itertuples(), 1):
            print(f"{i:3d}. company_id={row.company_id:>10s}  samples={row.sample_count:>6,d}")

        if len(eligible) > 20:
            print(f"... and {len(eligible)-20} more (see {args.output})")

        print("\n\nSample training commands:")
        print("-" * 60)
        for company_id in eligible.head(5)['company_id']:
            print(f"python train_autoencoder.py --mode entity --entity-id {company_id} \\")
            print(f"    --window {args.window} --train-start \"{args.start}\" --train-end \"{args.end}\"")
            print()

    needs_cluster = df[~df['sufficient_data']].copy()
    if len(needs_cluster) > 0:
        print(f"\n\nEntities needing CLUSTER models ({len(needs_cluster)} total):")
        print(f"These will use cluster fallback models (insufficient data)")
        print(f"Top 10 by sample count:")
        print(needs_cluster.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
