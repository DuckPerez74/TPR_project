import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque
import pandas as pd
from logging.handlers import RotatingFileHandler

from .context_builder import ContextBuilder
from .prompts import get_analysis_prompt
from .response_parser import ResponseParser


def _setup_llm_file_logger(enabled: bool = True) -> logging.Logger:
    """Setup a dedicated file logger for LLM operations."""
    logger = logging.getLogger('llm_file_logger')
    
    # Avoid adding multiple handlers if already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Only add file handler if file logging is enabled
    if enabled:
        # Log file in the llm folder
        log_path = Path(__file__).parent / 'llm.log'
        
        # Rotating file handler: 5MB max, keep 3 backups
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed format for debugging
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    else:
        # Add null handler to prevent "no handler" warnings
        logger.addHandler(logging.NullHandler())
    
    return logger


class LLMAnalyzer:
    def __init__(self, config_path: str = None):
        """
        Initialize LLM Analyzer.

        Args:
            config_path: Path to llm_config.json. If None, uses default location.
        """
        self.logger = logging.getLogger('wodle-tpr.llm')
        
        # Load main config to check enable_file_logging
        main_config = self._load_main_config()
        enable_file_logging = main_config.get('logging', {}).get('enable_file_logging', True)
        
        self.file_logger = _setup_llm_file_logger(enabled=enable_file_logging)

        if config_path is None:
            config_path = Path(__file__).parent / 'llm_config.json'

        self._log(f"=== LLM Analyzer Initializing ===")
        self._log(f"Config path: {config_path}")

        self.config = self._load_config(config_path)
        self.enabled = self.config.get('enabled', False)
        self.provider = self.config.get('provider', 'anthropic')
        self.model = self.config.get('model', 'claude-sonnet-4-5-20250929')
        self.trigger_window = self.config.get('trigger_on_window', 10)
        self.max_tokens = self.config.get('max_tokens_response', 1500)
        self.timeout = self.config.get('timeout_seconds', 30)
        self.temperature = self.config.get('temperature', 0.2)

        # Log configuration values
        self._log(f"Enabled: {self.enabled}")
        self._log(f"Provider: {self.provider}")
        self._log(f"Model: {self.model}")
        self._log(f"Trigger window: {self.trigger_window} min")
        self._log(f"Max tokens: {self.max_tokens}")
        self._log(f"Timeout: {self.timeout}s")
        self._log(f"Temperature: {self.temperature}")

        # Rate limiting configuration
        self.max_calls_per_hour = self.config.get('rate_limit', {}).get('max_calls_per_hour', 100)
        self.max_calls_per_day = self.config.get('rate_limit', {}).get('max_calls_per_day', 1000)
        self.max_retries = self.config.get('retry', {}).get('max_retries', 3)
        self.retry_backoff_base = self.config.get('retry', {}).get('backoff_seconds', 2)

        self._log(f"Rate limit: {self.max_calls_per_hour}/hour, {self.max_calls_per_day}/day")
        self._log(f"Retries: {self.max_retries}, backoff base: {self.retry_backoff_base}s")

        # Rate limiting state
        self._call_history = deque()
        self._daily_calls = 0
        self._daily_reset_time = datetime.now() + timedelta(days=1)

        self.context_builder = ContextBuilder()
        self.response_parser = ResponseParser()

        self._client = None
        
        self._log(f"=== LLM Analyzer Ready (enabled={self.enabled}) ===")

    def _log(self, message: str, level: str = 'info'):
        """Log to both standard logger and file logger."""
        log_func = getattr(self.logger, level, self.logger.info)
        file_log_func = getattr(self.file_logger, level, self.file_logger.info)
        log_func(message)
        file_log_func(message)

    def _load_main_config(self) -> dict:
        """Load main config.json to get global settings like enable_file_logging."""
        try:
            main_config_path = Path(__file__).parent.parent / 'config.json'
            with open(main_config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.file_logger.warning(f"Failed to load LLM config: {e}")
            return {'enabled': False}

    def _get_client(self):
        """Get or create LLM client based on provider."""
        if self._client is not None:
            return self._client

        api_key_env = self.config.get('api_key_env', 'ANTHROPIC_API_KEY')
        api_key = os.getenv(api_key_env)

        if not api_key:
            self._log(f"API key not found in environment variable: {api_key_env}", 'error')
            return None

        try:
            if self.provider == 'anthropic':
                from anthropic import Anthropic
                self._client = Anthropic(api_key=api_key)
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            return self._client
        except ImportError as e:
            self._log(f"LLM package not installed: {e}", 'error')
            return None
        except Exception as e:
            self._log(f"Failed to initialize LLM client: {e}", 'error')
            return None

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if call is allowed, False if rate limit exceeded
        """
        now = datetime.now()

        # Reset daily counter if needed
        if now >= self._daily_reset_time:
            self._daily_calls = 0
            self._daily_reset_time = now + timedelta(days=1)
            self._log("Daily LLM call counter reset")

        # Check daily limit
        if self._daily_calls >= self.max_calls_per_day:
            self._log(f"Daily rate limit exceeded: {self._daily_calls}/{self.max_calls_per_day}", 'warning')
            return False

        # Clean old entries (older than 1 hour)
        one_hour_ago = now - timedelta(hours=1)
        while self._call_history and self._call_history[0] < one_hour_ago:
            self._call_history.popleft()

        # Check hourly limit
        if len(self._call_history) >= self.max_calls_per_hour:
            self._log(f"Hourly rate limit exceeded: {len(self._call_history)}/{self.max_calls_per_hour}", 'warning')
            return False

        return True

    def _record_call(self):
        """Record a successful API call for rate limiting."""
        self._call_history.append(datetime.now())
        self._daily_calls += 1

    def should_analyze(self, anomaly_result: dict) -> bool:
        """
        Determine if LLM analysis should run for this anomaly.
        
        Triggers when risk_score >= min_risk_score (default 60%).
        Logs of the trigger_window (default 10 min) are sent to the LLM.
        """
        entity_id = anomaly_result.get('entity_id', 'unknown') if anomaly_result else 'none'
        self._log(f"LLM should_analyze called for entity {entity_id}")

        if not self.enabled:
            self._log(f"LLM disabled, skipping entity {entity_id}")
            return False

        if not anomaly_result:
            return False

        risk_score = anomaly_result.get('risk_score', 0.0)
        min_risk_score = self.config.get('min_risk_score_for_analysis', 0.6)

        self._log(
            f"LLM check for entity {entity_id}: risk={risk_score:.2%} (need >={min_risk_score:.2%})"
        )

        if risk_score < min_risk_score:
            self._log(f"LLM skipped for entity {entity_id}: risk too low ({risk_score:.2%} < {min_risk_score:.2%})")
            return False

        self._log(f"LLM TRIGGERED for entity {entity_id}! (risk={risk_score:.2%})")
        return True

    def analyze(self, entity_id: str, anomaly_result: dict,
                metrics: dict, entity_df: pd.DataFrame) -> dict:
        """
        Perform LLM analysis on anomaly alert.

        Args:
            entity_id: Entity identifier
            anomaly_result: Result from HierarchicalAnalyzer
            metrics: Current metrics snapshot
            entity_df: DataFrame with entity logs

        Returns:
            Analysis result dictionary
        """
        if not self.should_analyze(anomaly_result):
            return None

        # Check rate limit
        if not self._check_rate_limit():
            self._log(f"Rate limit exceeded, skipping LLM analysis for {entity_id}", 'warning')
            return self._create_error_result(entity_id, "Rate limit exceeded")

        start_time = datetime.now(timezone.utc)

        try:
            # Filter logs to only include the trigger window (10 minutes by default)
            # This ensures larger windows (30, 60 min) are not sent to the LLM
            if '@timestamp' in entity_df.columns:
                # start_time already has UTC timezone, so just convert to pd.Timestamp
                # Don't use tz='UTC' when the input already has tzinfo
                window_start = pd.Timestamp(start_time - timedelta(minutes=self.trigger_window))
                filtered_df = entity_df[entity_df['@timestamp'] >= window_start].copy()
                if filtered_df.empty:
                    self._log(f"No logs found in {self.trigger_window}min window for {entity_id}", 'warning')
                    filtered_df = entity_df  # Fallback to full df if no logs in window
            else:
                filtered_df = entity_df

            context = self.context_builder.build_context(
                entity_id=entity_id,
                anomaly_result=anomaly_result,
                metrics=metrics,
                entity_df=filtered_df,
                window_minutes=self.trigger_window
            )

            context_text = self.context_builder.format_for_prompt(context)
            prompt = get_analysis_prompt(context_text)

            self._log(f"Calling LLM for entity {entity_id} with {len(context.get('logs', []))} logs")
            llm_response = self._call_llm(prompt)

            if llm_response is None:
                self._log(f"LLM call returned None for entity {entity_id}", 'error')
                return self._create_error_result(entity_id, "LLM call failed")

            parsed = self.response_parser.parse(llm_response)

            elapsed_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            result = {
                'entity_id': entity_id,
                'timestamp': start_time.isoformat() + 'Z',
                'execution_time_ms': elapsed_ms,
                'model': self.model,
                'logs_analyzed': len(context.get('logs', [])),
                **parsed
            }

            self._log(
                f"LLM analysis complete for {entity_id}: "
                f"{parsed.get('classification')} ({parsed.get('confidence')}) in {elapsed_ms}ms"
            )

            return result

        except Exception as e:
            self._log(f"LLM analysis failed for {entity_id}: {e}", 'error')
            return self._create_error_result(entity_id, str(e))

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API with prompt, including retry logic with exponential backoff.

        Args:
            prompt: Complete prompt string

        Returns:
            LLM response text or None on failure
        """
        self._log("Getting LLM client...")
        client = self._get_client()
        if client is None:
            self._log("Failed to get LLM client", 'error')
            return None

        last_error = None
        for attempt in range(self.max_retries):
            self._log(f"LLM API call attempt {attempt + 1}/{self.max_retries}")
            try:
                if self.provider == 'anthropic':
                    # Ensure client has timeout configured
                    if self._client is None:
                        self._log("Client is None, cannot proceed", 'error')
                        return None

                    response = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system="You are a security analyst expert in API monitoring and threat detection.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    result = response.content[0].text
                else:
                    # OpenAI
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a security analyst expert in API monitoring and threat detection."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        timeout=self.timeout
                    )
                    result = response.choices[0].message.content

                # Record successful call
                self._record_call()
                self._log(f"LLM API call successful, response length: {len(result)} chars")
                return result

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Check if it's a retryable error
                retryable_errors = ['RateLimitError', 'TimeoutError', 'APIConnectionError', 'InternalServerError']
                is_retryable = any(err in error_type for err in retryable_errors)

                if attempt < self.max_retries - 1 and is_retryable:
                    backoff_time = self.retry_backoff_base ** (attempt + 1)
                    self._log(
                        f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {error_type}. "
                        f"Retrying in {backoff_time}s...",
                        'warning'
                    )
                    time.sleep(backoff_time)
                else:
                    self._log(f"LLM API call failed after {attempt + 1} attempts: {e}", 'error')
                    break

        return None

    def _create_error_result(self, entity_id: str, error: str) -> dict:
        """Create error result when analysis fails."""
        return {
            'entity_id': entity_id,
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'classification': 'Unknown',
            'threat_type': 'None',
            'user_operations': [],
            'explanation': f'Analysis failed: {error}',
            'recommended_actions': [],
            'confidence': 'Low',
            'error': True
        }
