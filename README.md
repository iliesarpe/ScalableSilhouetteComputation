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

## Usage

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import silhouette_scalable as ss

# Create a dataset with real cluster structure and cluster it
X, _ = make_blobs(n_samples=10_000, n_features=16, centers=8, random_state=0)
labels = KMeans(n_clusters=8, n_init=5, random_state=0).fit_predict(X)

# Per-point silhouette estimates — O(n) in the estimation step
result = ss.compute_local(X, labels, t=64, seed=0)
print(result["global_silhouette"])   # float in [-1, 1]
print(result["local_silhouette"])    # list of n per-point values

# Fast global estimate — evaluate only m << n points, much faster for large n
result = ss.compute_global(X, labels, m=500, t=64, seed=0)
print(result["global_silhouette"])

# Exact silhouette — O(n²), only practical for small datasets (n ≲ 5 000)
result = ss.compute_exact(X[:2000], labels[:2000])
print(result["global_silhouette"])
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
| `compute_local` | O(n) | global + per-point | you need per-point values |
| `compute_global(m=m)` | O(m) | global only | you only need the scalar, n is large |
| `compute_global()` | O(n) | global only | global only, same accuracy as local |
| `compute_uniform` | O(n) | global + per-point | uniform-sampling baseline |
| `compute_exact` | O(n²) | global + per-point | ground truth on small datasets |

### Distances

All functions accept a `distance` keyword:

```python
sa.compute_local(X, labels, distance="manhattan")
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

## Approximation guarantees

Given a clustering $\mathcal{C} = \{C_1, \dots, C_k\}$ of a dataset $V = \{e_1,\dots,e_n\}$, the silhouette of a point $e \in C$ is

$$
s(e) = \frac{b(e)-a(e)}{\max\{a(e), b(e)\}}, \qquad
a(e) = \frac{\sum_{e' \in C} d(e,e')}{|C|-1}, \qquad
b(e) = \min_{C_j \neq C}\frac{\sum_{e' \in C_j}d(e,e')}{|C_j|},
$$

and the average silhouette is $s(\mathcal{C}) = \frac{1}{n}\sum_{e\in V} s(e)$.

`compute_local` and `compute_global` implement the **PPS-weighted estimator** $\hat{s}_2$ from the paper:

$$\Pr\!\left[|\hat{s}_2 - s(\mathcal{C})| \le \tfrac{4\varepsilon}{1-\varepsilon}\right] > 1 - \delta$$

with $O\!\left(\frac{nk}{\varepsilon^2}\log\frac{nk}{\delta}\right)$ distance computations.

<p align="center">
  <img src="plots/dataEx.png" width="45%">
  <img src="plots/Splot.png" width="45%">
</p>

---

## Best-k selection example

```python
import silhouette_scalable as ss
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=5_000, centers=5, n_features=8, random_state=0)

for k in range(2, 10):
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
    score  = sa.compute_global(X, labels, m=300, seed=0)["global_silhouette"]
    print(f"k={k}  silhouette={score:.3f}")
```

A runnable version with plots is in [`examples/quickstart.py`](examples/quickstart.py).

---

## Research / HPC usage

The repository also contains the full research codebase used to produce the paper results, including HDF5-based experiment pipelines and an Apptainer container for HPC clusters.

### Build the C++ binary

**Linux (Ubuntu/Debian)**
```bash
sudo apt install g++ make libhdf5-dev nlohmann-json3-dev
cd cppCode && make
conda env create -f research/environment.yml && conda activate silhouetteEnv
```

**macOS** (requires [Homebrew](https://brew.sh))
```bash
brew install hdf5 libomp nlohmann-json
cd cppCode && make
conda env create -f research/environment.yml && conda activate silhouetteEnv
```

**Apptainer** (recommended for HPC / reproducibility)
```bash
apptainer build image.sif image.def   # build once from repo root
sbatch slurm_launcher.slurm           # compiles C++ on first run
```

### Reproducing paper results

The full experiment pipeline (HDF5 datasets, cluster scripts, result aggregation) is available in the tagged research release:
[`v1.0-paper`](https://github.com/iliesarpe/ScalableSilhouetteComputation/tree/v1.0-paper)
