import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST')
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD')

if not all([OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASSWORD]):
    raise ValueError("Missing required environment variables. Check .env file.")
