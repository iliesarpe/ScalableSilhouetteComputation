<!-- Improved compatibility of back to top link: See: https://github.com/pull/73 -->
<a name="Scalable and Distributed Silhouette Approximation Documentation"></a>

## Scalable and Distributed Silhouette Approximation 

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [General use](#general-use)
- [Datasets](#datasets)
- [Results](#results)
- [Example analyses](#example-analyses)

<!-- ABOUT THE PROJECT -->
## Overview

This repository provides the necessary scripts and code to reproduce the results of our work _Scalable and Distributed Silhouette Approximation._

Our code can be used to efficiently compute the (average, and local) [Silhouette](https://en.wikipedia.org/wiki/Silhouette_(clustering)) of a $k$-clustering under arbitrary metric distances.

Problem: Given a $k$-clustering $CL = \\{C_1, \dots, C_k\\}$ of a dataset $V=\\{e_1,\dots, e_n\\}$ (e.g., obtained with $K$-means) we provide efficient approximate methods to:
* estimate the (local) silhouette coefficient $s(e)$ of _each_ element in $e\in V$ with small approximation error;
* a suite of algorithms to estimate the (average) silhouette coefficient of the $k$-clustering $CL$ (see Table below);
* a parallel implementation of our methods.

Recall that the silhouette coefficient of an element $e\in V$ assigned to cluster $C\in CL$ is defined as

$$
s(e) = \frac{b(e)-a(e)}{\text{max} \\{{a(e), b(e)}\\} } \enspace,
$$
where
$$
a(e) = \frac{\sum_{e' \in C} d(e,e')}{|C|-1},
\qquad
b(e) = \min_{\substack{C_j \in CL,\\ C_j \neq C}}\frac{\sum_{e' \in C_j}d(e,e')}{|C_j|}\enspace.
$$

While the _average_ silhouette of a clustering $CL$ is defined as,

$$
s(CL) =  \frac{1}{n} \sum_{e\in V} s(e) \enspace.
$$

**Example**: 
- _Left_: Dataset $ V= \\{ e_1\dots,e_n \\} $ for $n=15$ with elements in the Euclidean plane $\mathbb{R}^2$. 
Different shapes represent $k=4$ different clusters, that is $\mathcal{C} = \\{C_1=\\{e_1,\dots,e_3,e_{14}\\}, C_2=\\{e_4,\dots,e_{10}\\},C_3=\\{e_{11},e_{12},e_{13}\\}, C_4=\\{e_{15}\\}\\}$.
- _Right_: silhouette $s(e)$ of the elements $e\in V$ (where the distance is the Euclidean distance): 
values are grouped by clusters and sorted. 
The dashed line represents the value of the _silhouette of the clustering_ $\mathcal{C}$, marked with $s(\mathcal{C})$.
<p align="center">
  <img src="plots/dataEx.png" width="45%">
  <img src="plots/Splot.png" width="45%">
</p>


**Table**: Our suite of algorithms for the estimation of the average silhouette coefficient.
| Function (C++)      | Estimator              | <div style="width:240px">Guarantees</div>                                                   | Distance-Computations                                                                                                                     | <div style="width:160px">Probability</div>|
|----------------|------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| ``ComputeEstimatorSubsample``  | $\hat{s}_1$ | $\|\hat{s}_1 - s(CL)\| \le \varepsilon$ | $O\!\left(\dfrac{n}{\varepsilon^2}\log\left(\dfrac{1}{\delta}\right)\right)$ | $> 1 - \delta$ |
| ``ApproximateSilhPPS``* | $\hat{s}_2$ | $\|\hat{s}_2 - s(CL)\|\le \dfrac{4\varepsilon}{1-\varepsilon}$ | ${O}\left(\dfrac{nk}{\varepsilon^2}\log\left(\dfrac{nk}{\delta}\right)\right)$ | $> 1 - \delta$ |
| ``ApproximateSilhPPS``* | $\hat{s}_3$ | $\|\hat{s}_3 - s(CL)\|\le \dfrac{4\varepsilon_1}{1-\varepsilon_1} + \varepsilon_2$ | ${O}\left(\left(n+\dfrac{m(\varepsilon_2,\delta_2)k}{\varepsilon_1^2}\right)\left(\log\left(\dfrac{nk}{\delta_1}\right)\right)\right)$ | $> (1-\delta_1)(1-\delta_2)$ |

\* Function ``ApproximateSilhPPS`` first computes $\hat{s}_2$ and if supplied with a vector of values $m_1,\dots,m_J$ for $m_i< n, i\in [J]$ it computes $\hat{s}_3$ for each such $m$.

If you rely on our code please consider citing our work
> *Author Names*  
> Conference/Journal, Year  
> [Read the paper](link-to-paper)

[Back to top](#table-of-contents)

## Installation

You can compile and use our code in two ways:  
1. **From source (Python + C++)**  
2. **Using Apptainer (recommended for reproducibility and HPC clusters)**

---

### Option 1: Direct C++ compilation + Python setup

#### Linux (Ubuntu/Debian)

```bash
sudo apt install g++ make libhdf5-dev nlohmann-json3-dev
```

#### macOS

```bash
brew install hdf5 libomp nlohmann-json
```

#### Build

```bash
git clone <ourrepo>
cd cppCode
make          # produces ./efficientSilh
```

The Makefile detects the platform automatically and selects the correct HDF5 and OpenMP paths.

#### Python environment

```bash
conda env create -f cppCode/environment.yml
conda activate silhouetteEnv
```

---

### Option 2: Apptainer

Requirements: [Apptainer](https://apptainer.org/)

`image.def` is located at the **repository root**. It installs the full C++ toolchain and Python environment inside the container but does **not** bake the source code in — the source is bind-mounted at runtime, so Python scripts can be edited without rebuilding the image.

```bash
# Build from the repository root
apptainer build image.sif image.def
```

On HPC clusters with SLURM, use the provided submission script:

```bash
# Submit from the repository root
sbatch slurm_launcher.slurm
```

The script compiles `efficientSilh` inside the container on the first run and caches the binary in `cppCode/`. Subsequent runs skip compilation.

[Back to top](#table-of-contents)

## General use <a name="general-use"></a>

The binary `efficientSilh` is driven by a JSON configuration file:

```bash
./efficientSilh config.json
```

The easiest way to understand the configuration format is to run the quickstart demo (see [Example analyses](#example-analyses)), which generates one config file per `k` value and prints the path so you can inspect it directly.

### Configuration reference

```json
{
  "mode": 4,
  "dataset":    "points.csv",
  "assignment": "labels.csv",
  "distance":   "euclidean",
  "k":          10,
  "t":          64,
  "delta":      0.01,
  "threads":    1,
  "seed":       100,
  "outfile":    "result.json"
}
```

| Field | Description |
|---|---|
| `mode` | Always `4` for standalone (general) use |
| `dataset` | CSV of floating-point features — no header, one point per row |
| `assignment` | CSV of integer cluster labels — one label per row, 0-indexed |
| `distance` | One of `euclidean`, `squared`, `manhattan`, `cosine`, `canberra` |
| `k` | Number of clusters |
| `t` | Sample size (number of PPS samples per cluster). Use this **or** `epsilon`: when possible **always** set `t` |
| `epsilon` | Approximation parameter; `t` is derived automatically from the sample-complexity bound |
| `delta` | Failure probability for the approximation guarantee |
| `threads` | Number of OpenMP threads |
| `seed` | RNG seed for reproducibility |
| `outfile` | Path where the JSON result is written |

The result JSON contains the estimated global silhouette (`global_silhouette`), per-point local silhouettes (`local_silhouette`), and runtime information.

The plots below are produced by the quickstart demo (see [Example analyses](#example-analyses)) on a synthetic 5-cluster dataset with $t=64$ samples per cluster:

<p align="center">
  <img src="plots/silhouette_distribution.png" width="90%"><br>
  <em>Per-cluster silhouette distributions for k ∈ {2,…,8}. At k=5 (the true number of clusters) all clusters show uniformly high silhouette values and the estimated average is clearly the largest.</em>
</p>

<p align="center">
  <img src="plots/silhouette_avg_vs_k.png" width="50%"><br>
  <em>Estimated average silhouette vs k. The peak at k=5 correctly identifies the true cluster count.</em>
</p>

[Back to top](#table-of-contents)

## Datasets

We provide details on how to download each dataset used in our experimental evaluation in the folder `cppCode/data`
1. The file `links.txt` provide the URLs to download each dataset
2. After downloading each dataset make sure to decompress each dataset, and if applicable, remap some characters in the ``txt`` file as specified in `links.txt`
3. After the processing use the script `cppCode/data/parseData.py` to produce the CSV file needed to go to the next step, an example of usage is the following:
```bash
# Save the metro dataset (from Metro.csv) in a new CSV format without headers

python parseData.py metro
ls real/
# should show the file real/metro.csv
```
You should do it for all datasets used in our experiments

4. Go to `cppCode/` and make sure the Python environment `silhouetteEnv` is active, then cluster each dataset using
```bash
python driver.py clust-real
```
The above command will generate a [HDF5](https://github.com/HDFGroup/hdf5) file in the folder `cppCode/data/real/clust/`, where each file contain all the clustering configurations used in our experiments for each dataset.


[Back to top](#table-of-contents)

## Results
To reproduce our results, first make sure that the downloading and processing of the datasets, has been done as described in [Datasets](#datasets).
Next go to `cppCode/` and make sure the Python environment `silhouetteEnv` and run:
```bash
# Compute the exact silhouette values of all points for each clustering configuration
python driver.py ex-silh-real
```

1. Reproducing the sequential results for local and global estimation
```bash
# Compute the local and global silhouette, comparing our methods and existing state-of-the-art approaches
python driver.py approx-silh-real-ex
```
2. Reproducing the parallel experiments 
```bash
# Perform parallel tests
python driver.py parallel-tests
```
3. Reproducing the results for the selection of the best parameter $k$ 
```bash
# Obtain clusterings
python driver.py clust-bestk

# Obtain exact values of the silhouette 
python driver.py ex-silh-bestk

# Obtain approximate values comparing PPS and UNI
python driver.py approx-silh-bestk
```


[Back to top](#table-of-contents)

## Example analyses <a name="example-analyses"></a>

### Quickstart: best-$k$ selection on a synthetic dataset

`cppCode/quickstart.py` is a self-contained demo that requires no external data download. It generates a 2000-point, 8-dimensional dataset with 5 true clusters (`sklearn.datasets.make_blobs`) and runs `ApproximateSilhPPS` across a range of $k$ values using only $t=64$ samples per cluster, producing two plots:

- **`silhouette_distribution.png`** — classic silhouette diagram: per-cluster distributions sorted by silhouette value, one panel per $k$
- **`silhouette_avg_vs_k.png`** — estimated average silhouette $\hat{s}(\mathcal{C})$ vs $k$, with the best $k$ highlighted

```bash
cd cppCode

# default sweep: k ∈ {2, 3, 4, 5, 6, 7, 8} — true best k is 5
python quickstart.py --run ./efficientSilh

# save plots directly to the repo plots/ folder
python quickstart.py --run ./efficientSilh --plots-dir ../plots

# custom k range
python quickstart.py --run ./efficientSilh --k-values 2 4 6 8 10
```

Running without `--run` writes only the JSON config files to `examples/out/` and prints their paths — useful for understanding the configuration format without needing the binary compiled.

```bash
# Print configs only (no binary required)
python quickstart.py
```

The generated `examples/out/blobs_config_k*.json` files show exactly how to structure the configuration for mode 4, and are the recommended starting point for adapting the tool to your own dataset.

[Back to top](#table-of-contents)
