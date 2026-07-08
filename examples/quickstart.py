"""
Quickstart example for the standalone (mode 4) run of efficientSilh.

Demonstrates best-k selection on a synthetic 2000-point dataset (8 dimensions,
5 true clusters generated with sklearn.datasets.make_blobs) by running
ApproximateSilhPPS for a range of k values and producing two plots:

  silhouette_distribution.png  — per-cluster silhouette distributions for each k
  silhouette_avg_vs_k.png      — estimated average silhouette score vs k

Usage (from the cppCode/ directory):
    python quickstart.py                              # print configs only
    python quickstart.py --run ./efficientSilh        # run + plot
    python quickstart.py --run ./efficientSilh --k-values 2 3 4 5 6 7 8
    python quickstart.py --run ./efficientSilh --plots-dir ../plots
"""
import argparse
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")   # non-interactive; must come before pyplot/seaborn import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _generate_blobs(outdir: str, seed: int) -> tuple[str, np.ndarray]:
    """Generate isotropic Gaussian blobs and write the points CSV."""
    points_path = os.path.join(outdir, "blobs_points.csv")
    X, _ = make_blobs(n_samples=2000, n_features=8, centers=5,
                      cluster_std=1.5, random_state=seed)
    np.savetxt(points_path, X, delimiter=",", fmt="%.6f")
    print(f"Wrote {points_path}  ({X.shape[0]} points, {X.shape[1]} dims, 5 true clusters)")
    return points_path, X


def _labels_for_k(outdir: str, X: np.ndarray, k: int, seed: int) -> tuple[str, np.ndarray]:
    labels_path = os.path.join(outdir, f"blobs_labels_k{k}.csv")
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    np.savetxt(labels_path, labels, fmt="%d")
    return labels_path, labels


def _write_config(outdir: str, points_path: str, labels_path: str,
                  k: int, t: int, seed: int) -> tuple[str, str]:
    result_path = os.path.join(outdir, f"blobs_result_k{k}.json")
    config = {
        "mode": 4,
        "dataset": points_path,
        "assignment": labels_path,
        "distance": "euclidean",
        "k": k,
        "t": t,
        "delta": 0.01,
        "threads": 1,
        "seed": seed,
        "outfile": result_path,
    }
    config_path = os.path.join(outdir, f"blobs_config_k{k}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path, result_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_silhouette_distributions(all_results: list[dict], plots_dir: str) -> str:
    """Classic silhouette diagram, one subplot per k.

    Points within each cluster are sorted by silhouette value and drawn as a
    horizontal-bar fill.  A dashed vertical line marks the estimated global mean.
    """
    sns.set_style("whitegrid")
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        k = res["k"]
        palette = sns.color_palette("viridis", n_colors=k)
        local_silh = np.array(res["local_silhouette"])
        labels = np.array(res["labels"])
        avg = res["global_silhouette"]
        gap = max(2, len(local_silh) // 60)

        y_lower = 0
        for ci in range(k):
            vals = np.sort(local_silh[labels == ci])
            y_upper = y_lower + len(vals)
            color = palette[ci % len(palette)]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, vals,
                facecolor=color, edgecolor=color, alpha=0.75,
            )
            ax.text(-0.96, y_lower + len(vals) / 2, str(ci + 1),
                    fontsize=7, va="center", color="dimgray")
            y_lower = y_upper + gap

        ax.axvline(x=avg, color="black", linestyle="--", linewidth=1.3, alpha=0.7)
        ax.set_xlim([-1, 1])
        ax.set_xlabel("Silhouette value $\\hat{s}(e)$", fontsize=10)
        ax.set_title(f"$k = {k}, \\hat{{s}}(\\mathcal{{C}}) = {avg:.3f}$", fontsize=11)
        ax.yaxis.set_visible(False)
        sns.despine(ax=ax, left=True)

    axes[0].set_ylabel("Points (sorted within cluster)", fontsize=10)
    fig.suptitle("Silhouette plots with estimated local values — synthetic blobs",
                 fontsize=12, y=0.98)
    plt.tight_layout()
    out = os.path.join(plots_dir, "silhouette_distribution.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved  {out}")
    return out


def plot_avg_vs_k(all_results: list[dict], plots_dir: str) -> str:
    """Estimated average silhouette score vs k — the standard best-k selection curve."""
    sns.set_style("whitegrid")
    k_values = [r["k"] for r in all_results]
    avgs = [r["global_silhouette"] for r in all_results]
    best_idx = int(np.argmax(avgs))
    best_k = k_values[best_idx]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(k_values, avgs, marker="o", linewidth=2,
            color="steelblue", markersize=7, label="$\\hat{s}(\\mathcal{C})$")
    ax.axvline(x=best_k, color="crimson", linestyle="--", linewidth=1.4, alpha=0.8,
               label=f"Best $k = {best_k}$  ($\\hat{{s}} = {avgs[best_idx]:.3f}$)")
    ax.set_xlabel("Number of clusters $k$", fontsize=13)
    ax.set_ylabel("Estimated silhouette $\\hat{s}(\\mathcal{C})$", fontsize=13)
    ax.set_title("Best-$k$ selection via silhouette score", fontsize=14)
    ax.set_xticks(k_values)
    ax.legend(fontsize=11)
    sns.despine(ax=ax)
    ax.spines['left'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    out = os.path.join(plots_dir, "silhouette_avg_vs_k.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved  {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="examples/out",
                        help="directory for CSVs, configs, and result files (default: examples/out)")
    parser.add_argument("--plots-dir", default=None,
                        help="directory for output PNG plots; defaults to --outdir")
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8],
                        help="k values to evaluate (default: 2 3 4 5 6 7 8)")
    parser.add_argument("--t", type=int, default=64,
                        help="PPS sample size per cluster (default: 64)")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--run", default=None,
                        help="path to compiled efficientSilh binary; "
                             "if given, runs it and produces plots")
    args = parser.parse_args()

    plots_dir = args.plots_dir if args.plots_dir is not None else args.outdir
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    points_path, X = _generate_blobs(args.outdir, args.seed)

    all_results: list[dict] = []

    for k in args.k_values:
        labels_path, labels = _labels_for_k(args.outdir, X, k, args.seed)
        config_path, result_path = _write_config(
            args.outdir, points_path, labels_path, k, args.t, args.seed)

        if args.run:
            print(f"Running k={k} …")
            subprocess.run([args.run, config_path], check=True)
            with open(result_path) as f:
                res = json.load(f)
            res["labels"] = labels.tolist()
            all_results.append(res)
            print(f"  k={k}:  global silhouette ≈ {res['global_silhouette']:.4f}"
                  f"  ({res['runtime_seconds']:.2f} s)")
        else:
            print(f"Config for k={k} written to {config_path}")
            print(f"  Run with:  ./efficientSilh {config_path}")

    if not args.run:
        print("\nPass --run ./efficientSilh to execute all configs and generate plots.")
        return

    print("\nGenerating plots …")
    dist_path = plot_silhouette_distributions(all_results, plots_dir)
    avg_path  = plot_avg_vs_k(all_results, plots_dir)

    best_k = args.k_values[int(np.argmax([r["global_silhouette"] for r in all_results]))]
    print(f"\nBest k by estimated silhouette: k = {best_k}")
    print(f"Plots:\n  {dist_path}\n  {avg_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Note on the dataset:
# make_blobs() generates 2000 points in 8 dimensions across 5 isotropic
# Gaussian clusters.  The true k=5 wins clearly at t=64 — a good smoke-test
# that the algorithm is working.  To try a real-world dataset instead:
#
#   from sklearn.datasets import load_wine
#   data = load_wine()
#   X = data.data   # 178 points, 13 features, 3 true classes (k=3 should win)
#
# or the handwritten digits (may need larger t due to close silhouette scores):
#
#   from sklearn.datasets import load_digits
#   X = load_digits().data   # 1797 points, 64 features, 10 classes
# ---------------------------------------------------------------------------
