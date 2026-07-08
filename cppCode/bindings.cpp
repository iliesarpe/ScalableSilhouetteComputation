#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <stdexcept>
#include "libs/algorithms.h"

namespace py = pybind11;
using namespace pybind11::literals;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int dist_from_string(const std::string& name)
{
    if (name == "euclidean")   return EUCLID;
    if (name == "sqeuclidean") return SQUARED;
    if (name == "manhattan")   return MANHATTAN;
    if (name == "cosine")      return COSINE;
    if (name == "canberra")    return CANBERRA;
    throw std::invalid_argument(
        "Unknown distance \"" + name + "\". "
        "Valid options: euclidean, sqeuclidean, manhattan, cosine, canberra.");
}

// Convert numpy arrays → internal C++ types.
// Labels are cast to int (int32) via forcecast; cluster ids are never > 2^31.
static std::pair<std::vector<Structures::Point<double>>, Structures::AlgorithmParameters>
build_inputs(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int,    py::array::c_style | py::array::forcecast> labels,
    int k, int t, double delta, const std::string& distance, int threads, long long seed)
{
    auto xbuf = X.request();
    auto lbuf = labels.request();

    if (xbuf.ndim != 2)
        throw std::invalid_argument("X must be a 2-D array.");
    if (lbuf.ndim != 1)
        throw std::invalid_argument("labels must be a 1-D array.");

    long long n = xbuf.shape[0];
    int       d = static_cast<int>(xbuf.shape[1]);

    if (lbuf.shape[0] != n)
        throw std::invalid_argument(
            "X has " + std::to_string(n) + " rows but labels has " +
            std::to_string(lbuf.shape[0]) + " entries.");
    if (n == 0)
        throw std::invalid_argument("X is empty.");

    const double* xp = static_cast<const double*>(xbuf.ptr);
    const int*    lp = static_cast<const int*>(lbuf.ptr);

    if (k <= 0)
        k = 1 + *std::max_element(lp, lp + n);

    for (long long i = 0; i < n; i++) {
        if (lp[i] < 0 || lp[i] >= k)
            throw std::invalid_argument(
                "labels[" + std::to_string(i) + "] = " + std::to_string(lp[i]) +
                " is out of range [0, " + std::to_string(k - 1) + "]. "
                "Cluster ids must be 0-indexed. Pass k explicitly if it should "
                "not be inferred from max(labels)+1.");
    }

    std::vector<Structures::Point<double>> pointset;
    pointset.reserve(n);
    for (long long i = 0; i < n; i++) {
        std::vector<double> coords(xp + i * d, xp + (i + 1) * d);
        Structures::Point<double> p(coords);
        p.setClust(lp[i]);
        pointset.push_back(std::move(p));
    }

    Structures::AlgorithmParameters params;
    params.k                = k;
    params.t                = t;
    params.delta            = delta;
    params.distType         = dist_from_string(distance);
    params.threads          = threads;
    params.seed             = seed;
    params.sampleSizeDouble = {};
    params.clusterAssignments =
        Structures::BuildClusterAssignments(std::vector<int>(lp, lp + n), k);

    return {std::move(pointset), std::move(params)};
}

// ---------------------------------------------------------------------------
// Binding functions
// ---------------------------------------------------------------------------

static py::dict py_approximate_pps(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int,    py::array::c_style | py::array::forcecast> labels,
    int k, int t, double delta, const std::string& distance, int threads, long long seed)
{
    auto [pointset, params] = build_inputs(X, labels, k, t, delta, distance, threads, seed);

    Structures::AlgorithmResult res;
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        py::gil_scoped_release release;
        Algorithms::ApproximateSilhPPS(pointset, params, res);
    }
    double wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    return py::dict(
        "global_silhouette"_a = res.approxSilh,
        "local_silhouette"_a  = res.silhApproxValues,
        "runtime_seconds"_a   = wall);
}

static py::dict py_approximate_uniform(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int,    py::array::c_style | py::array::forcecast> labels,
    int k, int t, double delta, const std::string& distance, int threads, long long seed)
{
    auto [pointset, params] = build_inputs(X, labels, k, t, delta, distance, threads, seed);

    Structures::AlgorithmResult res;
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        py::gil_scoped_release release;
        Algorithms::ApproximateSilhUniform(pointset, params, res);
    }
    double wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    return py::dict(
        "global_silhouette"_a = res.approxSilh,
        "local_silhouette"_a  = res.silhApproxValues,
        "runtime_seconds"_a   = wall);
}

static py::dict py_approximate_pps_global(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int,    py::array::c_style | py::array::forcecast> labels,
    long long m, int k, int t, double delta,
    const std::string& distance, int threads, long long seed)
{
    auto [pointset, params] = build_inputs(X, labels, k, t, delta, distance, threads, seed);
    params.globalOnly = true;
    params.globalM    = (m > 0 && m < (long long)pointset.size()) ? m : -1;

    Structures::AlgorithmResult res;
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        py::gil_scoped_release release;
        Algorithms::ApproximateSilhPPS(pointset, params, res);
    }
    double wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    return py::dict(
        "global_silhouette"_a = res.approxSilh,
        "runtime_seconds"_a   = wall);
}

static py::dict py_exact_silhouette(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int,    py::array::c_style | py::array::forcecast> labels,
    int k, const std::string& distance)
{
    // t/delta/threads/seed are irrelevant for exact computation
    auto [pointset, params] = build_inputs(X, labels, k, 1, 0.01, distance, 1, 0);

    std::vector<double> local;
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        py::gil_scoped_release release;
        local = Algorithms::ComputeExactSilhouettePoints(pointset, params.k, params.distType);
    }
    double wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    double global = 0.0;
    for (double v : local) global += v;
    if (!local.empty()) global /= static_cast<double>(local.size());

    return py::dict(
        "global_silhouette"_a = global,
        "local_silhouette"_a  = local,
        "runtime_seconds"_a   = wall);
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_silhouette_approx, m)
{
    m.doc() = "Efficient silhouette approximation — C++ core via pybind11";

    m.def("approximate_pps",
        &py_approximate_pps,
        py::arg("X"),
        py::arg("labels"),
        py::arg("k")        = -1,
        py::arg("t")        = 64,
        py::arg("delta")    = 0.01,
        py::arg("distance") = "euclidean",
        py::arg("threads")  = 1,
        py::arg("seed")     = 100LL,
        R"doc(
Estimate silhouette via PPS-weighted sampling (the main algorithm, ŝ₂).

Parameters
----------
X        : array-like, shape (n, d), float64
labels   : array-like, shape (n,), integer cluster ids in [0, k)
k        : number of clusters; inferred as max(labels)+1 when ≤ 0
t        : sample size per cluster
delta    : failure probability
distance : "euclidean" | "sqeuclidean" | "manhattan" | "cosine" | "canberra"
threads  : OpenMP thread count
seed     : RNG seed (reproducibility)

Returns
-------
dict
    global_silhouette : float  — estimated average silhouette score
    local_silhouette  : list[float]  — per-point estimates, length n
    runtime_seconds   : float
)doc");

    m.def("approximate_uniform",
        &py_approximate_uniform,
        py::arg("X"),
        py::arg("labels"),
        py::arg("k")        = -1,
        py::arg("t")        = 64,
        py::arg("delta")    = 0.01,
        py::arg("distance") = "euclidean",
        py::arg("threads")  = 1,
        py::arg("seed")     = 100LL,
        R"doc(
Estimate silhouette via uniform random sampling (baseline).

Same signature and return type as approximate_pps.
)doc");

    m.def("approximate_pps_global",
        &py_approximate_pps_global,
        py::arg("X"),
        py::arg("labels"),
        py::arg("m")        = -1LL,
        py::arg("k")        = -1,
        py::arg("t")        = 64,
        py::arg("delta")    = 0.01,
        py::arg("distance") = "euclidean",
        py::arg("threads")  = 1,
        py::arg("seed")     = 100LL,
        R"doc(
Estimate the global silhouette score via PPS-weighted sampling.

Faster than approximate_pps when only the scalar score is needed: evaluates
GetEstimateTerms for m points instead of all n. Set m < n for the fast path;
omit m (or set m >= n) to evaluate all n points without storing per-point values.

Returns
-------
dict
    global_silhouette : float
    runtime_seconds   : float
)doc");

    m.def("exact_silhouette",
        &py_exact_silhouette,
        py::arg("X"),
        py::arg("labels"),
        py::arg("k")        = -1,
        py::arg("distance") = "euclidean",
        R"doc(
Compute the exact silhouette score in O(n² · d) time.

Only practical for small datasets (n ≲ 5 000).

Parameters
----------
X        : array-like, shape (n, d), float64
labels   : array-like, shape (n,), integer cluster ids in [0, k)
k        : number of clusters; inferred as max(labels)+1 when ≤ 0
distance : "euclidean" | "sqeuclidean" | "manhattan" | "cosine" | "canberra"

Returns
-------
dict
    global_silhouette : float
    local_silhouette  : list[float], length n
    runtime_seconds   : float
)doc");
}
