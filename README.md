# silhouette-scalable

Scalable approximate silhouette scoring for k-clusterings under arbitrary metric distances.

> I. Sarpe, F. Altieri, A. Pietracaprina, G. Pucci, F. Vandin.
> *Scalable and Distributed Silhouette Approximation*, arXiv, 2026.
> [Read the paper](https://arxiv.org/abs/2607.01993)

---

## Install

```bash
pip install silhouette-scalable
```

Requires Python ≥ 3.9 and NumPy. Pre-compiled wheels are available for Linux x86_64 and macOS arm64/x86_64. No C++ compiler needed.


---

## Quickstart

No data download required. The example runs on a synthetic dataset generated on the fly.

```bash
pip install silhouette-scalable matplotlib seaborn scikit-learn
python examples/quickstart.py
```

This will:
1. Generate 2 000 points in 8 dimensions with 5 true clusters (`make_blobs`)
2. Run k-means and estimate the silhouette for `k = 2,...,8`
3. Save two plots to `examples/out/`:
   - `silhouette_distribution.png` — per-cluster silhouette distributions for each k
   - `silhouette_avg_vs_k.png` — average silhouette vs k (the peak identifies k = 5)

Optional arguments:
```bash
python examples/quickstart.py --k-values 2 4 6 8 10   # custom k range
python examples/quickstart.py --t 128                 # larger sample → more accurate estimates
python examples/quickstart.py --plots-dir ./plots     # custom output directory
```

---

## Usage

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import silhouette_scalable as ss

# Create a dataset with real cluster structure and cluster it
n_points = 50000
X, _ = make_blobs(n_samples=n_points, n_features=16, centers=8, random_state=0)
labels = KMeans(n_clusters=8, n_init=5, random_state=0).fit_predict(X)

# Per-point silhouette estimates — O(n) in the estimation step
result = ss.compute_local(X, labels, t=64, seed=0)
print(f"Global silhouette: {result['global_silhouette']:.4f}  ({result['runtime_seconds']:.2f}s)")
print(f"First 10 local estimates: {[round(v, 3) for v in result['local_silhouette'][:10]]}")

# Fast global estimate — evaluate only m << n points, much faster when n is extremely large
result = ss.compute_global(X, labels, m=500, t=64, seed=0)
print(f"Global (fast path, m=500): {result['global_silhouette']:.4f}  ({result['runtime_seconds']:.2f}s)")

# Exact silhouette — O(n^2), reduce n_points if this is too slow on your machine
result = ss.compute_exact(X, labels)
print(f"Exact global silhouette:   {result['global_silhouette']:.4f}  ({result['runtime_seconds']:.2f}s)")
```

### Return value

All functions return a `dict`. The keys present depend on the function:

| Key | Type | Present in |
|---|---|---|
| `global_silhouette` | `float` — average silhouette score in [-1, 1] | all functions |
| `local_silhouette` | `list[float]` of length n — per-point scores | `compute_local`, `compute_uniform`, `compute_exact` |
| `runtime_seconds` | `float` — wall-clock time of the C++ call | all functions |

### When to use which function

| Function | Cost | Returns | Use when |
|---|---|---|---|
| `compute_local` | $O(nkt)$ | global + per-point | you need per-point values |
| `compute_global(m=m)` | $O(mkt)$ | global only | you only need the scalar, n is large |
| `compute_global()` | $O(nkt)$ | global only | global only, same accuracy as local |
| `compute_uniform` | $O(nkt)$ | global + per-point | uniform-sampling baseline |
| `compute_exact` | $O(n^2)$ | global + per-point | ground truth on small datasets |

We hide $log(nk/\delta)$ factor for simplicity, check the paper for the exact complexities.

For per-point silhouette distributions and best-k selection plots, see [`examples/quickstart.py`](examples/quickstart.py).
`compute_global` scales to datasets with tens of millions of points on commodity hardware.

### Distances

All functions accept a `distance` keyword:

```python
ss.compute_local(X, labels, distance="manhattan")
```

Supported: `"euclidean"` (default), `"sqeuclidean"`, `"manhattan"`, `"cosine"`, `"canberra"`.

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `t` | `64` | PPS sample size per cluster — larger gives tighter estimates |
| `delta` | `0.01` | Failure probability for the approximation guarantee |
| `m` | `n` | (`compute_global` only) number of points to evaluate |
| `threads` | `1` | OpenMP thread count (Linux wheels only) |
| `seed` | `100` | RNG seed for reproducibility |
| `k` | auto | Number of clusters — inferred as `max(labels)+1` if not set |

---

## Best-k selection example

```python
import silhouette_scalable as ss
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=5_000, centers=5, n_features=8, random_state=0)

for k in range(2, 10):
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
    score  = ss.compute_global(X, labels, m=300, seed=0)["global_silhouette"]
    print(f"k={k}  silhouette={score:.3f}")
```

A runnable version with per-cluster silhouette distribution plots is in [`examples/quickstart.py`](examples/quickstart.py):

```bash
pip install silhouette-scalable matplotlib seaborn scikit-learn
python examples/quickstart.py
```

---

## Reproducing paper results

The full experiment pipeline (HDF5 datasets, cluster scripts, result aggregation) is available in the tagged research release: [`v1.0-paper`](https://github.com/iliesarpe/ScalableSilhouetteComputation/tree/v1.0-paper)
