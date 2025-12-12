import json
import os
import re
from pathlib import Path
from typing import Any


class ConfigLoader:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self._config = None
        self._load()

    def _load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            content = f.read()

        content = self._substitute_env_vars(content)
        self._config = json.loads(content)

    def _substitute_env_vars(self, content: str) -> str:
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                raise ValueError(f"Environment variable not set: {env_var}")
            return value

        return re.sub(pattern, replacer, content)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> dict:
        return self._config.copy()
