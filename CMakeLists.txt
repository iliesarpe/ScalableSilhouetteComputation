cmake_minimum_required(VERSION 3.18)
project(silhouette_approx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- pybind11 (injected as a build dep via pyproject.toml) -----------------
find_package(pybind11 CONFIG REQUIRED)

# ---- OpenMP (optional: without it #pragma omp lines are silently ignored) --
# Apple Clang does not ship OpenMP; probe Homebrew libomp first.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE _BREW_LIBOMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_BREW_LIBOMP)
        set(OpenMP_CXX_FLAGS           "-Xclang -fopenmp")
        set(OpenMP_CXX_LIB_NAMES      "omp")
        set(OpenMP_omp_LIBRARY        "${_BREW_LIBOMP}/lib/libomp.dylib")
        include_directories(SYSTEM    "${_BREW_LIBOMP}/include")
    endif()
endif()
find_package(OpenMP)

# ---- Extension module -------------------------------------------------------
set(CSRC ${CMAKE_SOURCE_DIR}/cppCode)

pybind11_add_module(_silhouette_approx
    ${CSRC}/bindings.cpp
    ${CSRC}/libs/algorithms.cpp
    ${CSRC}/libs/structures.cpp
)

target_include_directories(_silhouette_approx PRIVATE
    ${CSRC}/libs
    ${CSRC}/import      # vendored: nlohmann/json, HighFive (HDF5 wrapper — guarded, unused)
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_silhouette_approx PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the extension into the wheel's package directory.
# scikit-build-core maps this to silhouette_scalable/ inside the wheel.
install(TARGETS _silhouette_approx DESTINATION silhouette_scalable)
