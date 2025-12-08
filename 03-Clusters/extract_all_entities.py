import pandas as pd
from opensearchpy import OpenSearch
import urllib3
import sys

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
ES_HOST = "https://100.125.228.80:9200"
ES_USER = sys.argv[1] if len(sys.argv) > 1 else "admin"
ES_PASSWORD = sys.argv[2] if len(sys.argv) > 2 else ""

if not ES_PASSWORD:
    print("Usage: python extract_all_entities.py [username] [password]")
    print("Example: python extract_all_entities.py admin mypassword")
    sys.exit(1)

INDEX_PATTERN = "metrics-l1-60m"
OUTPUT_FILE = "all_entities.csv"

print(f"\nConnecting to OpenSearch at {ES_HOST}...")

# Connect to OpenSearch
try:
    client = OpenSearch(
        hosts=[ES_HOST],
        http_auth=(ES_USER, ES_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30
    )

    # Test connection
    info = client.info()
    print(f"[OK] Connected to OpenSearch cluster: {info['cluster_name']}")
    print(f"  Version: {info['version']['number']}")

except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    sys.exit(1)

# Extract unique company_id using aggregation
print(f"\nExtracting unique company_id from index '{INDEX_PATTERN}'...")

try:
    # Use aggregation to get all unique company_id
    query = {
        "size": 0,
        "aggs": {
            "unique_companies": {
                "terms": {
                    "field": "company_id.keyword",
                    "size": 10000  # Max companies to retrieve
                }
            }
        }
    }

    response = client.search(index=INDEX_PATTERN, body=query)

    # Extract company IDs from buckets
    buckets = response['aggregations']['unique_companies']['buckets']
    company_ids = [bucket['key'] for bucket in buckets]

    print(f"[OK] Found {len(company_ids)} unique company_id")

    if len(company_ids) == 0:
        print("\n[WARNING] No company_id found. Check if:")
        print("  1. Index exists and has data")
        print("  2. Field name is correct (company_id)")
        print("  3. You have permissions to access the index")
        sys.exit(1)

    # Create DataFrame and save to CSV
    df = pd.DataFrame({'company_id': sorted(company_ids)})
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[OK] Saved {len(company_ids)} company IDs to '{OUTPUT_FILE}'")
    print(f"\nFirst 10 companies:")
    print(df.head(10).to_string(index=False))

    if len(company_ids) >= 10000:
        print("\n[WARNING] Reached maximum limit of 10,000 companies.")
        print("  If you have more, increase the 'size' parameter in the aggregation.")

except Exception as e:
    print(f"[ERROR] Query failed: {e}")
    sys.exit(1)

print(f"\n[OK] Done! You can now use '{OUTPUT_FILE}' for training.")
print(f"  Rename it or copy to: 03-Clusters/cluster0_entities.csv")
