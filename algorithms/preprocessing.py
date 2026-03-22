from __future__ import annotations

import numpy as np


def standardize_features(features: np.ndarray) -> np.ndarray:
    # Standardize each column to zero mean and unit variance.
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds == 0] = 1.0
    return (features - means) / stds


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    # Normalize each column to [0, 1].
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return (data - min_val) / range_val


def z_score_normalize(data: np.ndarray) -> np.ndarray:
    # Apply z-score scaling to each column.
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


def run_pca(
    features_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Compute PCA decomposition and variance statistics.
    covariance_matrix = np.cov(features_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative_variance = np.cumsum(explained_variance_ratio)
    return eigenvalues, eigenvectors, explained_variance_ratio, cumulative_variance


def choose_min_components(cumulative_variance: np.ndarray, threshold: float) -> int:
    # Return minimum number of components needed to reach variance threshold.
    return int(np.argmax(cumulative_variance >= threshold) + 1)


def project_data(
    features_scaled: np.ndarray,
    eigenvectors: np.ndarray,
    components: int,
) -> np.ndarray:
    # Project standardized data onto selected principal components.
    return features_scaled @ eigenvectors[:, :components]
