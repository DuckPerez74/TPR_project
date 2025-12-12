import pandas as pd
from abc import ABC, abstractmethod


class MetricsCalculator(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def calculate(self, df: pd.DataFrame, window: int) -> dict:
        pass

    def _safe_divide(self, numerator, denominator, default=0.0):
        if denominator == 0:
            return default
        return numerator / denominator

    def _safe_percentile(self, series: pd.Series, percentile: int):
        if series.empty:
            return 0.0
        return float(series.quantile(percentile / 100))
