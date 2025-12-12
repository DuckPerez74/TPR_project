import numpy as np
import pandas as pd


def calculate_entropy(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    return float(entropy)


def calculate_gini(series: pd.Series) -> float:
    if series.empty or len(series) == 0:
        return 0.0

    sorted_values = np.sort(series.values)
    n = len(sorted_values)
    total = np.sum(sorted_values)

    if total == 0 or n == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * total) - (n + 1) / n

    return float(gini)
