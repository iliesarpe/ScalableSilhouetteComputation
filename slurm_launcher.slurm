#!/bin/bash

#SBATCH --job-name=silh-approx
#SBATCH --output logs/RUN_%j.txt
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=00-01:59
#SBATCH --cpus-per-task=1

# SLURM_SUBMIT_DIR is set by the scheduler to the directory where sbatch was
# called.  Submit from the repository root:  sbatch parsing/run_silh.slurm
REPO_ROOT="${SLURM_SUBMIT_DIR}"
SIF="${REPO_ROOT}/image.sif"           # built with:  apptainer build image.sif image.def
CPPCODE="${REPO_ROOT}/cppCode"
container_home="/usr/src/silhouette"   # mount point inside the container

cd "${REPO_ROOT}"
mkdir -p logs

# ---------------------------------------------------------------------------
# Compile step (first run only)
#
# The image contains the compiler, HDF5, and nlohmann/json but NOT the source.
# We bind-mount cppCode/ into the container and run make; the binary is written
# back to the host at cppCode/efficientSilh and reused on every subsequent run.
# ---------------------------------------------------------------------------
if [ ! -f "${CPPCODE}/efficientSilh" ]; then
    echo "Compiling efficientSilh inside container (one-time)..."
    apptainer exec --fakeroot \
        -B "${CPPCODE}/:${container_home}:rw" \
        --pwd "${container_home}" \
        "${SIF}" make
fi

# ---------------------------------------------------------------------------
# Run step
#
# Same bind-mount: cppCode/ overlays /usr/src/silhouette/ so any edit to a
# Python script (quickstart.py, driver.py, ...) takes effect immediately
# without rebuilding the image, exactly as in the espresso pattern.
# The compiled binary is at ./efficientSilh relative to --pwd.
# ---------------------------------------------------------------------------

# --- Quickstart demo: best-k on digits, produces PDF plots ------------------
apptainer exec --fakeroot \
    -B "${CPPCODE}/:${container_home}:rw" \
    --pwd "${container_home}" \
    "${SIF}" \
    python quickstart.py --run ./efficientSilh --k-values 2 4 6 8 10

# --- Full experiment pipeline (uncomment the relevant step) -----------------
#apptainer exec --fakeroot -B "${CPPCODE}/:${container_home}:rw" --pwd "${container_home}" "${SIF}" python driver.py clust-real
#apptainer exec --fakeroot -B "${CPPCODE}/:${container_home}:rw" --pwd "${container_home}" "${SIF}" python driver.py ex-silh-real
#apptainer exec --fakeroot -B "${CPPCODE}/:${container_home}:rw" --pwd "${container_home}" "${SIF}" python driver.py approx-silh-real-ex
#apptainer exec --fakeroot -B "${CPPCODE}/:${container_home}:rw" --pwd "${container_home}" "${SIF}" python driver.py parallel-tests
#apptainer exec --fakeroot -B "${CPPCODE}/:${container_home}:rw" --pwd "${container_home}" "${SIF}" python driver.py clust-bestk
#apptainer exec --fakeroot -B "${CPPCODE}/:${container_home}:rw" --pwd "${container_home}" "${SIF}" python driver.py ex-silh-bestk
#apptainer exec --fakeroot -B "${CPPCODE}/:${container_home}:rw" --pwd "${container_home}" "${SIF}" python driver.py approx-silh-bestk
