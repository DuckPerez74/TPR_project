import pandas as pd
from abc import ABC, abstractmethod


class MetricsCalculator(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def calculate(self, df: pd.DataFrame, window: int) -> dict:
        pass
