from opensearchpy import OpenSearch
import threading
from urllib.parse import urlparse
import warnings

warnings.filterwarnings('ignore', message='.*Unverified HTTPS request.*')
warnings.filterwarnings('ignore', message='.*SSL.*', category=Warning)


class OpenSearchClient:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, config: dict):
        if OpenSearchClient._instance is not None:
            raise RuntimeError("Use get_instance() instead")

        opensearch_config = config.get('opensearch', {})
        verify_certs = opensearch_config.get('verify_certs', True)

        # SSL certificate verification warning suppressed
        # User is aware of security implications when verify_certs=False

        # Parse host to extract base URL and path prefix
        host = opensearch_config['host']
        parsed = urlparse(host)

        # Reconstruct base host without path
        base_host = f"{parsed.scheme}://{parsed.netloc}"
        url_prefix = parsed.path.rstrip('/') if parsed.path and parsed.path != '/' else None

        client_params = {
            'hosts': [base_host],
            'http_auth': (opensearch_config['username'], opensearch_config['password']),
            'verify_certs': verify_certs,
            'ssl_show_warn': False,  # Suppress SSL warnings
            'timeout': opensearch_config.get('timeout', 30),
            'max_retries': opensearch_config.get('max_retries', 3),
            'retry_on_timeout': True
        }

        # Add url_prefix if path exists
        if url_prefix:
            client_params['url_prefix'] = url_prefix

        self.client = OpenSearch(**client_params)

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
