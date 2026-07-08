"""
silhouette_scalable — scalable approximate silhouette scoring for k-clusterings.
"""

from __future__ import annotations

import numpy as np
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("silhouette-scalable")
except PackageNotFoundError:
    __version__ = "unknown"

from ._silhouette_approx import approximate_pps as _pps
from ._silhouette_approx import approximate_pps_global as _pps_global
from ._silhouette_approx import approximate_uniform as _uniform
from ._silhouette_approx import exact_silhouette as _exact

VALID_DISTANCES = frozenset({"euclidean", "sqeuclidean", "manhattan", "cosine", "canberra"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare(X, labels):
    X = np.asarray(X, dtype=np.float64, order="C")
    labels = np.asarray(labels, dtype=np.int32, order="C")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}.")
    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but labels has {labels.shape[0]} entries."
        )
    if X.shape[0] == 0:
        raise ValueError("X is empty.")
    return X, labels


def _check_distance(distance):
    if distance not in VALID_DISTANCES:
        raise ValueError(
            f"Unknown distance '{distance}'. "
            f"Valid options: {sorted(VALID_DISTANCES)}."
        )


def _check_sampling_params(t, delta, threads):
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}.")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_local(
    X,
    labels,
    *,
    distance: str = "euclidean",
    t: int = 64,
    delta: float = 0.01,
    threads: int = 1,
    seed: int = 100,
    k: int = -1,
) -> dict:
    """Estimate per-point silhouette scores via PPS-weighted sampling.

    Evaluates all n points — O(n) in the estimation step. Use this when you
    need the per-point ``local_silhouette`` values. For just the global scalar
    on large datasets, prefer :func:`compute_global` with ``m < n``.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Data points, converted to float64.
    labels : array-like of shape (n,)
        Integer cluster assignments in [0, k). Converted to int32.
    distance : str, default "euclidean"
        One of: "euclidean", "sqeuclidean", "manhattan", "cosine", "canberra".
    t : int, default 64
        Sample size per cluster. Larger values give tighter estimates.
    delta : float, default 0.01
        Failure probability for the approximation guarantee.
    threads : int, default 1
        Number of OpenMP threads. Requires the package was built with OpenMP.
    seed : int, default 100
        RNG seed for reproducibility.
    k : int, default -1
        Number of clusters. Inferred as ``max(labels) + 1`` when <= 0.

    Returns
    -------
    dict
        global_silhouette : float
            Estimated average silhouette score in [-1, 1].
        local_silhouette : list[float]
            Per-point estimates, length n.
        runtime_seconds : float
    """
    X, labels = _prepare(X, labels)
    _check_distance(distance)
    _check_sampling_params(t, delta, threads)
    return _pps(X, labels, k=k, t=t, delta=delta,
                distance=distance, threads=threads, seed=seed)


def compute_global(
    X,
    labels,
    *,
    distance: str = "euclidean",
    t: int = 64,
    delta: float = 0.01,
    threads: int = 1,
    seed: int = 100,
    k: int = -1,
    m: int = -1,
) -> dict:
    """Estimate the global silhouette score via PPS-weighted sampling.

    When ``m < n``, only ``m`` randomly chosen points are evaluated in the
    estimation step — O(m) instead of O(n). This can be orders of magnitude
    faster on large datasets when only the scalar score is needed.

    When ``m`` is omitted or ``m >= n``, all n points are evaluated (same
    accuracy as :func:`compute_local`) but per-point values are not returned.

    Parameters
    ----------
    X : array-like of shape (n, d)
    labels : array-like of shape (n,)
    distance : str, default "euclidean"
    t : int, default 64
        Sample size per cluster for the PPS sampling step.
    delta : float, default 0.01
    threads : int, default 1
    seed : int, default 100
    k : int, default -1
        Number of clusters. Inferred as ``max(labels) + 1`` when <= 0.
    m : int, default -1
        Number of points to evaluate. Set ``m < n`` for the fast path.
        Omit or set to -1 / >= n to evaluate all points.

    Returns
    -------
    dict
        global_silhouette : float
        runtime_seconds : float
    """
    X, labels = _prepare(X, labels)
    _check_distance(distance)
    _check_sampling_params(t, delta, threads)
    n = len(X)
    m_val = int(m) if (m is not None and 0 < m < n) else -1
    return _pps_global(X, labels, m=m_val, k=k, t=t, delta=delta,
                       distance=distance, threads=threads, seed=seed)


def compute_uniform(
    X,
    labels,
    *,
    distance: str = "euclidean",
    t: int = 64,
    delta: float = 0.01,
    threads: int = 1,
    seed: int = 100,
    k: int = -1,
) -> dict:
    """Estimate the silhouette score via uniform random sampling (baseline).

    Identical signature and return type to :func:`compute`.
    """
    X, labels = _prepare(X, labels)
    _check_distance(distance)
    _check_sampling_params(t, delta, threads)
    return _uniform(X, labels, k=k, t=t, delta=delta,
                    distance=distance, threads=threads, seed=seed)


def compute_exact(
    X,
    labels,
    *,
    distance: str = "euclidean",
    k: int = -1,
) -> dict:
    """Compute the exact silhouette score in O(n^2 * d) time.

    Only practical for small datasets (n <= ~5 000).

    Parameters
    ----------
    X : array-like of shape (n, d)
    labels : array-like of shape (n,)
        Integer cluster assignments in [0, k).
    distance : str, default "euclidean"
        One of: "euclidean", "sqeuclidean", "manhattan", "cosine", "canberra".
    k : int, default -1
        Number of clusters. Inferred as ``max(labels) + 1`` when <= 0.

    Returns
    -------
    dict
        global_silhouette : float
        local_silhouette : list[float], length n
        runtime_seconds : float
    """
    X, labels = _prepare(X, labels)
    _check_distance(distance)
    return _exact(X, labels, k=k, distance=distance)


compute = compute_local  # backward-compatible alias

__all__ = [
    "__version__",
    "VALID_DISTANCES",
    "compute_local",
    "compute_global",
    "compute_uniform",
    "compute_exact",
    "compute",  # alias for compute_local
]
