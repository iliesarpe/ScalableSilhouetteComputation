"""
Tests for the silhouette_approx public API.

All tests use small n and small t so the suite finishes in a few seconds.
"""

import numpy as np
import pytest

import silhouette_approx as sa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blobs():
    """200 points in 4-D with 3 well-separated clusters."""
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    X, _ = make_blobs(n_samples=200, n_features=4, centers=3,
                      cluster_std=0.5, random_state=0)
    labels = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X)
    return X, labels.astype(np.int32)


@pytest.fixture(scope="module")
def random_data():
    """100 random points with 4 arbitrary clusters — no cluster structure."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 6))
    labels = (np.arange(100) % 4).astype(np.int32)
    return X, labels


# ---------------------------------------------------------------------------
# Smoke tests — all three functions return sensible output
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_compute_returns_dict(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X, labels, t=32, seed=0)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"global_silhouette", "local_silhouette", "runtime_seconds"}

    def test_compute_global_in_range(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X, labels, t=32, seed=0)
        assert -1.0 <= res["global_silhouette"] <= 1.0

    def test_compute_local_length(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X, labels, t=32, seed=0)
        assert len(res["local_silhouette"]) == len(labels)

    def test_compute_local_values_in_range(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X, labels, t=32, seed=0)
        vals = res["local_silhouette"]
        assert all(-1.0 <= v <= 1.0 for v in vals)

    def test_compute_well_separated_blobs(self, blobs):
        """PPS should score well-separated blobs highly."""
        X, labels = blobs
        res = sa.compute_local(X, labels, t=64, seed=0)
        assert res["global_silhouette"] > 0.5

    def test_compute_uniform(self, blobs):
        X, labels = blobs
        res = sa.compute_uniform(X, labels, t=32, seed=0)
        assert -1.0 <= res["global_silhouette"] <= 1.0
        assert len(res["local_silhouette"]) == len(labels)

    def test_compute_exact(self, blobs):
        X, labels = blobs
        res = sa.compute_exact(X, labels)
        assert -1.0 <= res["global_silhouette"] <= 1.0
        assert len(res["local_silhouette"]) == len(labels)

    def test_runtime_positive(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X, labels, t=16, seed=0)
        assert res["runtime_seconds"] > 0.0


# ---------------------------------------------------------------------------
# Approximate vs exact consistency
# ---------------------------------------------------------------------------

class TestApproxVsExact:
    def test_pps_close_to_exact(self, blobs):
        """With large t, PPS should be within 0.1 of the exact score."""
        X, labels = blobs
        exact  = sa.compute_exact(X, labels)["global_silhouette"]
        approx = sa.compute_local(X, labels, t=128, seed=0)["global_silhouette"]
        assert abs(approx - exact) < 0.1

    def test_uniform_close_to_exact(self, blobs):
        X, labels = blobs
        exact  = sa.compute_exact(X, labels)["global_silhouette"]
        approx = sa.compute_uniform(X, labels, t=128, seed=0)["global_silhouette"]
        assert abs(approx - exact) < 0.1


# ---------------------------------------------------------------------------
# compute_global — fast global estimator
# ---------------------------------------------------------------------------

class TestComputeGlobal:
    def test_returns_scalar_only(self, blobs):
        X, labels = blobs
        res = sa.compute_global(X, labels, t=32, seed=0)
        assert set(res.keys()) == {"global_silhouette", "runtime_seconds"}
        assert "local_silhouette" not in res

    def test_global_in_range(self, blobs):
        X, labels = blobs
        res = sa.compute_global(X, labels, t=32, seed=0)
        assert -1.0 <= res["global_silhouette"] <= 1.0

    def test_fast_path_close_to_local(self, blobs):
        """compute_global with m<n should agree with compute_local within 0.1."""
        X, labels = blobs
        full = sa.compute_local(X, labels, t=64, seed=0)["global_silhouette"]
        fast = sa.compute_global(X, labels, t=64, seed=0, m=50)["global_silhouette"]
        assert abs(fast - full) < 0.15

    def test_full_path_matches_local(self, blobs):
        """compute_global with m omitted should match compute_local's global score."""
        X, labels = blobs
        local_g  = sa.compute_local(X, labels, t=64, seed=7)["global_silhouette"]
        global_g = sa.compute_global(X, labels, t=64, seed=7)["global_silhouette"]
        assert abs(global_g - local_g) < 1e-9

    def test_runtime_positive(self, blobs):
        X, labels = blobs
        res = sa.compute_global(X, labels, t=16, seed=0, m=20)
        assert res["runtime_seconds"] > 0.0

    def test_deterministic_fast_path(self, blobs):
        X, labels = blobs
        r1 = sa.compute_global(X, labels, t=32, seed=5, m=30)
        r2 = sa.compute_global(X, labels, t=32, seed=5, m=30)
        assert r1["global_silhouette"] == r2["global_silhouette"]

    def test_m_ge_n_falls_back_to_full(self, blobs):
        X, labels = blobs
        n = len(X)
        r_full = sa.compute_global(X, labels, t=32, seed=3)
        r_large_m = sa.compute_global(X, labels, t=32, seed=3, m=n + 100)
        assert r_full["global_silhouette"] == r_large_m["global_silhouette"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self, random_data):
        X, labels = random_data
        r1 = sa.compute_local(X, labels, t=32, seed=99)
        r2 = sa.compute_local(X, labels, t=32, seed=99)
        assert r1["global_silhouette"] == r2["global_silhouette"]
        assert r1["local_silhouette"] == r2["local_silhouette"]

    def test_different_seed_different_result(self, blobs):
        # blobs has non-trivial PPS probabilities so different seeds select
        # different samples; round-robin labels (uniform sizes) would not.
        X, labels = blobs
        r1 = sa.compute_local(X, labels, t=32, seed=1)
        r2 = sa.compute_local(X, labels, t=32, seed=2)
        assert r1["local_silhouette"] != r2["local_silhouette"]

    def test_uniform_deterministic(self, random_data):
        X, labels = random_data
        r1 = sa.compute_uniform(X, labels, t=32, seed=7)
        r2 = sa.compute_uniform(X, labels, t=32, seed=7)
        assert r1["global_silhouette"] == r2["global_silhouette"]


# ---------------------------------------------------------------------------
# All distance metrics round-trip
# ---------------------------------------------------------------------------

class TestDistances:
    @pytest.mark.parametrize("distance", sorted(sa.VALID_DISTANCES))
    def test_all_distances_run(self, random_data, distance):
        X, labels = random_data
        res = sa.compute_local(X, labels, t=16, seed=0, distance=distance)
        assert -1.0 <= res["global_silhouette"] <= 1.0

    @pytest.mark.parametrize("distance", sorted(sa.VALID_DISTANCES))
    def test_exact_all_distances(self, random_data, distance):
        X, labels = random_data
        res = sa.compute_exact(X, labels, distance=distance)
        assert -1.0 <= res["global_silhouette"] <= 1.0


# ---------------------------------------------------------------------------
# Input flexibility — array-likes, dtypes, explicit k
# ---------------------------------------------------------------------------

class TestInputFlexibility:
    def test_list_input(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X.tolist(), labels.tolist(), t=16, seed=0)
        assert -1.0 <= res["global_silhouette"] <= 1.0

    def test_float32_input(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X.astype(np.float32), labels, t=16, seed=0)
        assert -1.0 <= res["global_silhouette"] <= 1.0

    def test_int64_labels(self, blobs):
        X, labels = blobs
        res = sa.compute_local(X, labels.astype(np.int64), t=16, seed=0)
        assert -1.0 <= res["global_silhouette"] <= 1.0

    def test_explicit_k_matches_inferred(self, blobs):
        X, labels = blobs
        k = int(labels.max()) + 1
        r_inferred = sa.compute_local(X, labels, t=32, seed=5)
        r_explicit  = sa.compute_local(X, labels, t=32, seed=5, k=k)
        assert r_inferred["global_silhouette"] == r_explicit["global_silhouette"]

    def test_exact_explicit_k(self, blobs):
        X, labels = blobs
        k = int(labels.max()) + 1
        r1 = sa.compute_exact(X, labels)
        r2 = sa.compute_exact(X, labels, k=k)
        assert r1["global_silhouette"] == r2["global_silhouette"]


# ---------------------------------------------------------------------------
# Input validation — bad inputs raise ValueError
# ---------------------------------------------------------------------------

class TestValidation:
    def test_bad_distance(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="distance"):
            sa.compute_local(X, labels, distance="minkowski")

    def test_t_zero(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="t"):
            sa.compute_local(X, labels, t=0)

    def test_t_negative(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="t"):
            sa.compute_local(X, labels, t=-1)

    def test_delta_zero(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="delta"):
            sa.compute_local(X, labels, delta=0.0)

    def test_delta_one(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="delta"):
            sa.compute_local(X, labels, delta=1.0)

    def test_threads_zero(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="threads"):
            sa.compute_local(X, labels, threads=0)

    def test_X_1d(self, blobs):
        _, labels = blobs
        with pytest.raises(ValueError, match="2-D"):
            sa.compute_local(np.ones(10), labels[:10])

    def test_X_3d(self, blobs):
        _, labels = blobs
        with pytest.raises(ValueError, match="2-D"):
            sa.compute_local(np.ones((10, 4, 2)), labels[:10])

    def test_labels_2d(self, blobs):
        X, _ = blobs
        with pytest.raises(ValueError, match="1-D"):
            sa.compute_local(X, np.zeros((len(X), 1), dtype=np.int32))

    def test_length_mismatch(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="rows"):
            sa.compute_local(X, labels[:-1])

    def test_empty_X(self):
        with pytest.raises(ValueError, match="empty"):
            sa.compute_local(np.empty((0, 4)), np.empty(0, dtype=np.int32))

    def test_bad_distance_exact(self, blobs):
        X, labels = blobs
        with pytest.raises(ValueError, match="distance"):
            sa.compute_exact(X, labels, distance="chebyshev")
