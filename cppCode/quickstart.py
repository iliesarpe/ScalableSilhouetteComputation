"""
Quickstart example for the standalone (mode 4) run of efficientSilh.

Demonstrates best-k selection on the scikit-learn "digits" dataset (1797 points,
64-dim, 10 digit classes) by running ApproximateSilhPPS for a range of k values
and producing two publication-quality plots:

  silhouette_distribution.pdf  — per-cluster silhouette distributions for each k
  silhouette_avg_vs_k.pdf      — estimated average silhouette score vs k

Usage (from the cppCode/ directory):
    python quickstart.py                              # print configs only
    python quickstart.py --run ./efficientSilh        # run + plot
    python quickstart.py --run ./efficientSilh --k-values 2 4 6 8 10
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
from sklearn.datasets import load_digits


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_digits_once(outdir: str) -> tuple[str, np.ndarray]:
    points_path = os.path.join(outdir, "digits_points.csv")
    digits = load_digits()
    X = digits.data
    if not os.path.exists(points_path):
        np.savetxt(points_path, X, delimiter=",", fmt="%.6f")
        print(f"Wrote {points_path}  ({X.shape[0]} points, {X.shape[1]} dims)")
    else:
        print(f"Reusing existing {points_path}")
    return points_path, X


def _labels_for_k(outdir: str, X: np.ndarray, k: int, seed: int) -> tuple[str, np.ndarray]:
    labels_path = os.path.join(outdir, f"digits_labels_k{k}.csv")
    if os.path.exists(labels_path):
        labels = np.loadtxt(labels_path, dtype=int)
    else:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        np.savetxt(labels_path, labels, fmt="%d")
    return labels_path, labels


def _write_config(outdir: str, points_path: str, labels_path: str,
                  k: int, t: int, seed: int) -> tuple[str, str]:
    result_path = os.path.join(outdir, f"digits_result_k{k}.json")
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
    config_path = os.path.join(outdir, f"digits_config_k{k}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path, result_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_silhouette_distributions(all_results: list[dict], outdir: str) -> str:
    """Classic silhouette diagram, one subplot per k.

    Points within each cluster are sorted by silhouette value and drawn as a
    horizontal-bar fill.  A dashed vertical line marks the estimated global mean.
    """
    sns.set_style("whitegrid")
    palette = sns.color_palette("tab10")
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        k = res["k"]
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

        ax.axvline(x=avg, color="black", linestyle="--", linewidth=1.3, alpha=0.85)
        ax.set_xlim([-1, 1])
        ax.set_xlabel("Silhouette value $s(e)$", fontsize=10)
        ax.set_title(f"$k = {k}$\n$\\hat{{s}} = {avg:.3f}$", fontsize=11)
        ax.yaxis.set_visible(False)
        sns.despine(ax=ax, left=True)

    axes[0].set_ylabel("Points (sorted within cluster)", fontsize=10)
    fig.suptitle("Estimated local silhouette distributions — digits dataset",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, "silhouette_distribution.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {out}")
    return out


def plot_avg_vs_k(all_results: list[dict], outdir: str) -> str:
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
    ax.set_ylabel("Estimated avg. silhouette $\\hat{s}(\\mathcal{C})$", fontsize=13)
    ax.set_title("Best-$k$ selection via silhouette score", fontsize=14)
    ax.set_xticks(k_values)
    ax.legend(fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = os.path.join(outdir, "silhouette_avg_vs_k.pdf")
    fig.savefig(out, bbox_inches="tight")
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
                        help="directory for CSVs, configs, results, and plots")
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6, 8, 10],
                        help="k values to evaluate (default: 2 4 6 8 10)")
    parser.add_argument("--t", type=int, default=64,
                        help="PPS sample size per cluster (default: 64)")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--run", default=None,
                        help="path to compiled efficientSilh binary; "
                             "if given, runs it and produces plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    points_path, X = _load_digits_once(args.outdir)

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
            res["labels"] = labels.tolist()   # attach for plotting
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
    dist_path = plot_silhouette_distributions(all_results, args.outdir)
    avg_path  = plot_avg_vs_k(all_results, args.outdir)

    best_k = args.k_values[int(np.argmax([r["global_silhouette"] for r in all_results]))]
    print(f"\nBest k by estimated silhouette: k = {best_k}")
    print(f"Plots:\n  {dist_path}\n  {avg_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Note on the dataset:
# load_digits() ships inside scikit-learn's package data — nothing is fetched
# from the network.  For a bigger/more realistic dataset, swap the loader for:
#
#   from sklearn.datasets import fetch_openml
#   mnist = fetch_openml("mnist_784", version=1, cache=True)
#   X = mnist.data.to_numpy()
#
# fetch_openml caches to ~/scikit_learn_data by default.
# ---------------------------------------------------------------------------
