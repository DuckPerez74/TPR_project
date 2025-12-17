"""
Global constants for the TPR anomaly detection system.

This module centralizes all feature orders and constants used across training,
detection, and metrics calculation to ensure consistency.
"""

L1_FEATURE_ORDER = [
    'total_requests', 'mean_requests_per_minute', 'max_requests_per_minute',
    'std_requests_per_minute', 'cv_request_rate', 'peak_to_average_ratio', 'burst_score',
    'pct_2xx_responses', 'pct_3xx_responses', 'pct_4xx_responses', 'pct_5xx_responses',
    'error_rate', 'critical_error_rate', 'status_code_entropy', 'unique_status_codes',
    'mean_response_time', 'std_response_time', 'p50_response_time', 'p75_response_time',
    'p90_response_time', 'p95_response_time', 'p99_response_time',
    'pct_slow_requests', 'pct_very_slow_requests',
    'unique_source_ips', 'mean_requests_per_ip', 'max_requests_single_ip',
    'gini_ip_distribution', 'ip_concentration_top10pct',
    'unique_api_modules', 'module_entropy', 'top_module_percentage', 'module_switching_frequency',
    'unique_routes', 'route_entropy', 'top5_routes_percentage',
    'mean_response_size', 'std_response_size', 'max_response_size', 'min_response_size',
    'unique_user_agents', 'user_agent_entropy', 'bot_like_ua_percentage',
    'unique_http_methods', 'get_request_ratio', 'post_request_ratio'
]

L2_USER_FEATURES = [
    'request_count', 'error_rate', 'success_rate', 'auth_failure_ratio',
    'mean_response_time', 'p95_response_time', 'mean_response_size',
    'deviation_from_personal_baseline', 'data_download_spike', 'error_rate_deviation',
    'privilege_endpoint_ratio', 'sensitive_data_access_rate', 'config_modification_attempts',
    'backup_access_indicator', 'export_endpoint_usage', 'bulk_operation_ratio',
    'working_hours_deviation', 'velocity_impossibility', 'unique_routes_accessed',
    'route_diversity'
]

L2_ROUTE_FEATURES = [
    'request_count', 'error_rate', 'success_rate', 'auth_failure_ratio',
    'mean_response_time', 'p95_response_time', 'mean_response_size',
    'unique_users', 'unique_source_ips', 'admin_usage_rate',
    'manager_usage_rate', 'technician_usage_rate', 'account_type_diversity',
    'unusual_account_access'
]

L2_SUPPORTED_DIMENSIONS = ['user', 'route']
SLOW_REQUEST_SECONDS = 1.0
VERY_SLOW_REQUEST_SECONDS = 5.0
VELOCITY_IMPOSSIBILITY_KMH = 800  # Speed of commercial airplane
MODIFICATION_METHODS = ['PUT', 'POST', 'PATCH', 'DELETE']
THRESHOLD_PERCENTILE = 'p99'
THRESHOLD_FALLBACK = 0.1
L1_SCORE_MULTIPLIER = 2.0
L2_NORM_INPUT_MIN = -0.3
L2_NORM_INPUT_MAX = 0.5
