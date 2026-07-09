"""
Quickstart example for silhouette-scalable.

Demonstrates best-k selection on a synthetic dataset (make_blobs, 5 true
clusters, 8 dimensions) by running compute_local for a range of k values
and producing two plots:

  silhouette_distribution.png  — per-cluster silhouette distributions for each k
  silhouette_avg_vs_k.png      — estimated average silhouette score vs k

Usage:
    python quickstart.py
    python quickstart.py --k-values 2 3 4 5 6 7 8 --t 128
    python quickstart.py --plots-dir ./plots
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive; must come before pyplot/seaborn import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import silhouette_scalable as ss


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_silhouette_distributions(all_results: list[dict], plots_dir: str) -> str:
    """Classic silhouette diagram: one subplot per k, bars sorted within each cluster."""
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

    fig.text(0.01, 0.5, "Points (sorted within cluster)",
             va="center", rotation="vertical", fontsize=10)
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
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    parser.add_argument("--plots-dir", default=default_out,
                        help="directory for output PNG plots (default: examples/out)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8],
                        help="k values to evaluate (default: 2 3 4 5 6 7 8)")
    parser.add_argument("--t", type=int, default=64,
                        help="PPS sample size per cluster (default: 64)")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print("Generating synthetic dataset (make_blobs, 2000 points, 8 dims, 5 true clusters)…")
    X, _ = make_blobs(n_samples=2000, n_features=8, centers=5,
                      cluster_std=1.5, random_state=args.seed)

    all_results: list[dict] = []

    for k in args.k_values:
        labels = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(X)
        res = ss.compute_local(X, labels, t=args.t, seed=args.seed)
        res["k"] = k
        res["labels"] = labels.tolist()
        all_results.append(res)
        print(f"  k={k}:  global silhouette ≈ {res['global_silhouette']:.4f}"
              f"  ({res['runtime_seconds']:.3f} s)")

    best_k = args.k_values[int(np.argmax([r["global_silhouette"] for r in all_results]))]
    print(f"\nBest k by estimated silhouette: k = {best_k}")

    print("\nGenerating plots…")
    dist_path = plot_silhouette_distributions(all_results, args.plots_dir)
    avg_path  = plot_avg_vs_k(all_results, args.plots_dir)
    print(f"Plots:\n  {dist_path}\n  {avg_path}")


if __name__ == "__main__":
    main()
