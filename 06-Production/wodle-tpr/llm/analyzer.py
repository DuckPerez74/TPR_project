import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import pandas as pd

from .context_builder import ContextBuilder
from .prompts import get_analysis_prompt
from .response_parser import ResponseParser


class LLMAnalyzer:
    def __init__(self, config_path: str = None):
        """
        Initialize LLM Analyzer.

        Args:
            config_path: Path to llm_config.json. If None, uses default location.
        """
        self.logger = logging.getLogger('wodle-tpr.llm')

        if config_path is None:
            config_path = Path(__file__).parent / 'llm_config.json'

        self.config = self._load_config(config_path)
        self.enabled = self.config.get('enabled', False)
        self.provider = self.config.get('provider', 'anthropic')
        self.model = self.config.get('model', 'claude-sonnet-4-5-20250929')
        self.trigger_window = self.config.get('trigger_on_window', 10)
        self.max_tokens = self.config.get('max_tokens_response', 1500)
        self.timeout = self.config.get('timeout_seconds', 30)
        self.temperature = self.config.get('temperature', 0.2)

        # Rate limiting configuration
        self.max_calls_per_hour = self.config.get('rate_limit', {}).get('max_calls_per_hour', 100)
        self.max_calls_per_day = self.config.get('rate_limit', {}).get('max_calls_per_day', 1000)
        self.max_retries = self.config.get('retry', {}).get('max_retries', 3)
        self.retry_backoff_base = self.config.get('retry', {}).get('backoff_seconds', 2)

        # Rate limiting state
        self._call_history = deque()
        self._daily_calls = 0
        self._daily_reset_time = datetime.now() + timedelta(days=1)

        self.context_builder = ContextBuilder()
        self.response_parser = ResponseParser()

        self._client = None

    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load LLM config: {e}")
            return {'enabled': False}

    def _get_client(self):
        """Get or create LLM client based on provider."""
        if self._client is not None:
            return self._client

        api_key_env = self.config.get('api_key_env', 'ANTHROPIC_API_KEY')
        api_key = os.getenv(api_key_env)

        if not api_key:
            self.logger.error(f"API key not found in environment variable: {api_key_env}")
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
            self.logger.error(f"LLM package not installed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
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
            self.logger.info("Daily LLM call counter reset")

        # Check daily limit
        if self._daily_calls >= self.max_calls_per_day:
            self.logger.warning(f"Daily rate limit exceeded: {self._daily_calls}/{self.max_calls_per_day}")
            return False

        # Clean old entries (older than 1 hour)
        one_hour_ago = now - timedelta(hours=1)
        while self._call_history and self._call_history[0] < one_hour_ago:
            self._call_history.popleft()

        # Check hourly limit
        if len(self._call_history) >= self.max_calls_per_hour:
            self.logger.warning(f"Hourly rate limit exceeded: {len(self._call_history)}/{self.max_calls_per_hour}")
            return False

        return True

    def _record_call(self):
        """Record a successful API call for rate limiting."""
        self._call_history.append(datetime.now())
        self._daily_calls += 1

    def should_analyze(self, anomaly_result: dict) -> bool:
        if not self.enabled:
            return False

        if not anomaly_result:
            return False

        selected_window = anomaly_result.get('selected_window', 60)
        if selected_window != self.trigger_window:
            return False

        # Check risk_score threshold (default: 60% = 0.6)
        min_risk_score = self.config.get('min_risk_score_for_analysis', 0.6)
        risk_score = anomaly_result.get('risk_score', 0.0)
        
        if risk_score < min_risk_score:
            self.logger.debug(
                f"Skipping LLM analysis: risk_score {risk_score:.2%} < threshold {min_risk_score:.2%}"
            )
            return False

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
            self.logger.warning(f"Rate limit exceeded, skipping LLM analysis for {entity_id}")
            return self._create_error_result(entity_id, "Rate limit exceeded")

        start_time = datetime.utcnow()

        try:
            # Filter logs to only include the trigger window (10 minutes by default)
            # This ensures larger windows (30, 60 min) are not sent to the LLM
            if '@timestamp' in entity_df.columns:
                window_start = start_time - timedelta(minutes=self.trigger_window)
                filtered_df = entity_df[entity_df['@timestamp'] >= window_start].copy()
                if filtered_df.empty:
                    self.logger.warning(f"No logs found in {self.trigger_window}min window for {entity_id}")
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

            llm_response = self._call_llm(prompt)

            if llm_response is None:
                return self._create_error_result(entity_id, "LLM call failed")

            parsed = self.response_parser.parse(llm_response)

            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            result = {
                'entity_id': entity_id,
                'timestamp': start_time.isoformat() + 'Z',
                'execution_time_ms': elapsed_ms,
                'model': self.model,
                'logs_analyzed': len(context.get('logs', [])),
                **parsed
            }

            self.logger.info(
                f"LLM analysis complete for {entity_id}: "
                f"{parsed.get('classification')} ({parsed.get('confidence')})"
            )

            return result

        except Exception as e:
            self.logger.error(f"LLM analysis failed for {entity_id}: {e}")
            return self._create_error_result(entity_id, str(e))

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API with prompt, including retry logic with exponential backoff.

        Args:
            prompt: Complete prompt string

        Returns:
            LLM response text or None on failure
        """
        client = self._get_client()
        if client is None:
            return None

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == 'anthropic':
                    # Anthropic SDK doesn't support timeout directly in the call
                    # We'll use a wrapper approach
                    import httpx
                    timeout_config = httpx.Timeout(self.timeout, connect=10.0)

                    if self._client is None or not hasattr(self._client, '_client'):
                        # Reinitialize with timeout
                        from anthropic import Anthropic
                        api_key = os.getenv(self.config.get('api_key_env', 'ANTHROPIC_API_KEY'))
                        self._client = Anthropic(
                            api_key=api_key,
                            timeout=timeout_config
                        )

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
                return result

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Check if it's a retryable error
                retryable_errors = ['RateLimitError', 'TimeoutError', 'APIConnectionError', 'InternalServerError']
                is_retryable = any(err in error_type for err in retryable_errors)

                if attempt < self.max_retries - 1 and is_retryable:
                    backoff_time = self.retry_backoff_base ** (attempt + 1)
                    self.logger.warning(
                        f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {error_type}. "
                        f"Retrying in {backoff_time}s..."
                    )
                    time.sleep(backoff_time)
                else:
                    self.logger.error(f"LLM API call failed after {attempt + 1} attempts: {e}")
                    break

        return None

    def _create_error_result(self, entity_id: str, error: str) -> dict:
        """Create error result when analysis fails."""
        return {
            'entity_id': entity_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'classification': 'Unknown',
            'threat_type': 'None',
            'user_operations': [],
            'explanation': f'Analysis failed: {error}',
            'recommended_actions': [],
            'confidence': 'Low',
            'error': True
        }
