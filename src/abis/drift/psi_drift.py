import numpy as np
from typing import List


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    Measures how much a distribution changed.
    """
    expected = expected.astype(float)
    actual = actual.astype(float)

    # Use quantile-based bins from expected
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    # Avoid divide by zero
    eps = 1e-6
    expected_perc = np.clip(expected_perc, eps, 1)
    actual_perc = np.clip(actual_perc, eps, 1)

    value = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(value)


def psi_drift_score(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Average PSI across all features.
    reference/current: arrays of shape (n_samples, n_features)
    """
    scores: List[float] = []
    for j in range(reference.shape[1]):
        scores.append(psi(reference[:, j], current[:, j], bins=bins))
    return float(np.mean(scores))
