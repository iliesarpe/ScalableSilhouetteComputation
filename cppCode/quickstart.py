"""
Quickstart example for the standalone (mode 4) run of efficientSilh.

What this does:
  1. Loads the classic scikit-learn "digits" dataset (1797 8x8 grayscale digit
     images, flattened to 64-dim vectors, 10 classes) -- this ships inside
     scikit-learn, so nothing is downloaded from the network. If you want a
     dataset that IS fetched over the network, see the --fetch-openml note
     at the bottom of this file.
  2. Clusters it with k-means (k=10, matching the number of digit classes).
  3. Writes the two flat files mode-4 expects: a points CSV and a labels CSV.
  4. Writes a ready-to-run JSON config for ./efficientSilh.
  5. Optionally calls the compiled binary directly and reports the result.

Usage:
    python examples/quickstart_digits.py
    python examples/quickstart_digits.py --run ../efficientSilh
"""
import argparse
import json
import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans


def build_inputs(outdir: str, k: int, seed: int) -> tuple[str, str, int]:
    """Writes points.csv / labels.csv into `outdir` (skips work if they already
    exist), returning their paths plus the number of points."""
    points_path = os.path.join(outdir, "digits_points.csv")
    labels_path = os.path.join(outdir, "digits_labels.csv")

    if os.path.exists(points_path) and os.path.exists(labels_path):
        print(f"Found existing {points_path} / {labels_path}, reusing them.")
        n = sum(1 for _ in open(points_path))
        return points_path, labels_path, n

    os.makedirs(outdir, exist_ok=True)
    print("Loading digits dataset (bundled with scikit-learn, no download)...")
    digits = load_digits()
    X = digits.data  # shape (1797, 64), already flattened

    print(f"Clustering {X.shape[0]} points into k={k} clusters with k-means...")
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)

    # mode-4 points file: plain CSV, no header, no index column
    np.savetxt(points_path, X, delimiter=",", fmt="%.6f")
    # mode-4 assignment file: one 0-indexed cluster id per line
    np.savetxt(labels_path, labels, fmt="%d")

    print(f"Wrote {points_path} ({X.shape[0]} points, {X.shape[1]} dims)")
    print(f"Wrote {labels_path} ({len(labels)} labels)")
    return points_path, labels_path, X.shape[0]


def build_config(points_path: str, labels_path: str, k: int, outdir: str) -> str:
    config = {
        "mode": 4,
        "dataset": points_path,
        "assignment": labels_path,
        "distance": "euclidean",
        "k": k,
		"t": 5,
        "delta": 0.01,        # fix the sample size yourself, see README
        "threads": 4,
        "seed": 100,
        "outfile": os.path.join(outdir, "digits_result.json"),
    }
    config_path = os.path.join(outdir, "digits_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="examples/out", help="where to write points/labels/config/result")
    parser.add_argument("--k", type=int, default=10, help="number of clusters for k-means")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--run", default=None, help="path to the compiled efficientSilh binary; if given, runs it")
    args = parser.parse_args()

    points_path, labels_path, n = build_inputs(args.outdir, args.k, args.seed)
    config_path = build_config(points_path, labels_path, args.k, args.outdir)

    print()
    print("Config ready. Run it with:")
    print(f"  ./efficientSilh {config_path}")

    if args.run:
        import subprocess
        print(f"\nRunning {args.run} {config_path} ...")
        subprocess.run([args.run, config_path], check=True)
        with open(os.path.join(args.outdir, "digits_result.json")) as f:
            result = json.load(f)
        print(f"\nGlobal silhouette estimate: {result['global_silhouette']:.4f}")
        print(f"Runtime: {result['runtime_seconds']:.3f}s over n={result['n']} points")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Note on "downloading" a dataset:
# load_digits() ships inside scikit-learn's package data, so nothing is
# fetched over the network here. If you specifically want an example that
# exercises a real download-if-missing path (e.g. for a bigger/more
# realistic demo), swap the loader for:
#
#   from sklearn.datasets import fetch_openml
#   digits = fetch_openml("mnist_784", version=1, cache=True)
#
# fetch_openml caches to ~/scikit_learn_data by default, so reruns are free
# after the first download -- functionally the same "if not existing" logic
# this script already applies to points.csv/labels.csv.
# ---------------------------------------------------------------------------
