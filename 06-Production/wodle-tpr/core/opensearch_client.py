from opensearchpy import OpenSearch
import threading


class OpenSearchClient:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, config: dict):
        if OpenSearchClient._instance is not None:
            raise RuntimeError("Use get_instance() instead")

        opensearch_config = config.get('opensearch', {})
        verify_certs = opensearch_config.get('verify_certs', True)

        if not verify_certs:
            import sys
            print("WARNING: SSL certificate verification is disabled. This is insecure for production.", file=sys.stderr)

        self.client = OpenSearch(
            hosts=[opensearch_config['host']],
            http_auth=(opensearch_config['username'], opensearch_config['password']),
            verify_certs=verify_certs,
            ssl_show_warn=verify_certs,
            timeout=opensearch_config.get('timeout', 30),
            max_retries=opensearch_config.get('max_retries', 3),
            retry_on_timeout=True
        )

    @classmethod
    def get_instance(cls, config: dict = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if config is None:
                        raise ValueError("Config required for first initialization")
                    cls._instance = cls(config)
        return cls._instance.client

    @classmethod
    def reset_instance(cls):
        cls._instance = None
