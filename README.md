<!-- Improved compatibility of back to top link: See: https://github.com/pull/73 -->
<a name="Scalable and Distributed Silhouette Approximation Documentation"></a>

## Scalable and Distributed Silhouette Approximation 

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Results](#results)
- [Example analyses](#Example-analyses)
- [General use](#General-use)

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
The dashed line represents the value of the \emph{silhouette of the clustering} $\mathcal{C}$, marked with $s(\mathcal{C})$.
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
2. **Using Apptainer (recommended for reproducibility and ease of setup)**

**Note!** Our code was tested on a system running Ubuntu 20.04.

---

### Option 1: direct C++ compilation + Python setup
Requirements:  
- **C++17 or later** (`g++` compiler)  

Steps:  
```bash
# Clone repository
git clone <ourrepo>
cd cppCode

# Build C++ executable
make
```

#### Python environment
##### Option A — Using Conda (recommended)

Create an environment from the YAML file:

```bash
# Create from environment.yml (will use the python version listed there if present)
conda env create -f cppCode/environment.yml

# Activate
conda activate silhouetteEnv
```

#### Option B — Using Python
```bash
# Create requirements.txt from a YAML that has lines like "- pkg==x.y.z" under "dependencies:"
grep '^- ' environment.yml | sed 's/- //' > requirements.txt
```



### Option 2: Apptainer
Requirements:  
- [Apptainer](https://apptainer.org/)

```bash
# Build the container
apptainer build image.sif image.def
```

Now you can use your `image.sif` as an exacutable


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
#Compute the exact silhouette values of all points for each clustering configuration
python driver.py ex-silh-real.py
```

1. Reproducing the sequential results for local and global estimation
```bash
# Compute the local and global silhouette, comparing our methods and existing state-of-the-art approaches
python driver.py approx-silh-real-ex.py
```
2. Reproducing the parallel experiments 
```bash
# Perform parallel tests
python driver.py parallel-tests.py
```
3. Reproducing the results for the selection of the best parameter $k$ 
```bash
# Obtain clusterings
python driver.py clust-bestk.py

# Obtain exact values of the silhouette 
python driver.py ex-silh-bestk.py

# Obtain approximate values comparing PPS and UNI
python driver.py approx-silh-bestk.py
```


[Back to top](#table-of-contents)

## Example analysies <a name="Example-analyses"></a> 

## General use <a name="General-use"></a> 
Here we will specify how to use our tool for datasets outside the ones the ones tested in our experiments.


[Back to top](#table-of-contents)
