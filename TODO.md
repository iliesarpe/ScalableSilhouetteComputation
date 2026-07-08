# TODO — pip packaging with pybind11

Tasks are ordered by dependency. 1→2 must be sequential; 3 and 4 can run in parallel once 1–2 are done; 5 and 6 follow after 3–4; 7 is last.

---

- [x] **1. Decouple HDF5 from the core algorithm code**
  Guard all HDF5 includes and I/O functions in `structures.h/cpp` and `main.cpp`
  with `#ifdef USE_HDF5`. When the flag is absent the build succeeds without
  HDF5 installed. This isolates the research workflow (modes 1–3) from the
  pybind11 interface. Without this every `pip install` would require
  `libhdf5-dev` on the user's machine.

- [x] **2. Write pybind11 bindings for the core algorithms**
  Create `csrc/bindings.cpp` wrapping `ApproximateSilhPPS`,
  `ApproximateSilhUniform`, and `ComputeExactSilhouettePoints` via pybind11.
  Accept numpy arrays (`py::array_t`) directly — no CSV files, no subprocess,
  no temp files. Use pybind11's buffer protocol for zero-copy numpy interop.
  Distance type accepted as a string (`"euclidean"`, `"manhattan"`, …) and
  mapped internally to the existing int codes.

- [x] **3. Restructure repository layout**
  Reorganize into:
  ```
  src/silhouette_approx/   ← installable Python package
  csrc/                    ← C++ source (renamed from cppCode/)
  tests/
  examples/                ← quickstart.py moves here
  research/                ← driver.py, clustData.py (not installed)
  ```

- [x] **4. Write `CMakeLists.txt` and `pyproject.toml`**
  - `CMakeLists.txt`: find OpenMP + pybind11, compile `bindings.cpp` +
    `algorithms.cpp` + `structures.cpp` into a Python extension
    (`_silhouette_approx.so`). Build backend: `scikit-build-core`.
  - `pyproject.toml`: package metadata, runtime dep `numpy`, optional extras
    `matplotlib`/`seaborn` for plotting.
  - The existing `Makefile` stays for the research workflow; CMake is only for
    the pip build.

- [x] **5. Write the public Python API (`src/silhouette_approx/__init__.py`)**
  Expose a clean numpy-native interface:
  ```python
  compute(X, labels, *, distance="euclidean", t=64, delta=0.01,
          threads=1, seed=100) -> dict
  ```
  `X` is `(n, d)` array-like, `labels` is `(n,)` integer array-like.
  Returns `{"global_silhouette": float, "local_silhouette": [...],
  "runtime_seconds": float}`.
  Also expose distance name constants and `__version__`.

- [x] **6. Write tests (`tests/test_compute.py`)**
  Cover with pytest:
  - Smoke test: `make_blobs` → `compute()` returns a float in `[-1, 1]` and
    `local_silhouette` has length `n`.
  - All distance types round-trip without error.
  - Input validation raises on wrong shapes or unknown distance strings.
  - Determinism: same seed → same result.
  Keep `n` and `t` small so tests run fast in CI.

- [x] **7. Set up GitHub Actions CI with `cibuildwheel`**

---

## Open extensions

- **Custom / precomputed distances** — useful for graph-based metrics (shortest paths,
  diffusion distances) and any domain-specific distance that does not fit the five
  built-in metrics. The right API is likely a precomputed `(n, n)` distance matrix
  passed to a new `compute_from_distmatrix(D, labels)` function, which avoids the
  Python-callback-per-pair GIL bottleneck. Defer until there is clear user demand;
  track as a separate issue.
  - `release.yml`: on `v*` tag, build wheels for `manylinux_2_28_x86_64`,
    `macosx_arm64`, `macosx_x86_64`; upload to PyPI via trusted publisher.
  - `ci.yml`: on PRs, build + run tests only (no upload).
  - Ensure OpenMP is available in cibuildwheel containers
    (`libomp-dev` or equivalent in `before-all`).
