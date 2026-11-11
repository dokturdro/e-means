"""Compare e-means with Lloyd's k-means on SIFT1M.

Follows the paper's protocol: both methods start from the same random sample of training
vectors, Lloyd runs 25 iterations, and quality is the MSE on held-out vectors.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker

from clustering import e_means, e_means_recipe, lloyd, mse
from data import load_sift1m

LLOYD_ITERATIONS = 25
LLOYD_REPORTED_ITERATIONS = (1, 2, 3, 5, 10, 25)
E_MEANS_EPOCHS = (1, 2, 3, 4, 5)
LLOYD_COLOR, E_MEANS_COLOR, REFERENCE_COLOR = "#2a78d6", "#eb6834", "#8a8984"


class Result(NamedTuple):
    seconds: float
    train_mse: float
    held_out_mse: float


def elapsed(start: float) -> float:
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_lloyd(train: torch.Tensor, held_out: torch.Tensor, init: torch.Tensor) -> list[Result]:
    """One result per iteration, timed cumulatively without the evaluation in between."""
    results = []
    seconds = 0.0
    start = time.perf_counter()
    for centroids in lloyd(train, init, LLOYD_ITERATIONS):
        seconds += elapsed(start)
        results.append(Result(seconds, mse(train, centroids), mse(held_out, centroids)))
        start = time.perf_counter()
    return results


def run_e_means(train: torch.Tensor, held_out: torch.Tensor, init: torch.Tensor, seed: int) -> list[Result]:
    """One independent run per budget in E_MEANS_EPOCHS: the recipe and schedule depend on it."""
    results = []
    for epochs in E_MEANS_EPOCHS:
        start = time.perf_counter()
        centroids = e_means(train, init, epochs, seed)
        seconds = elapsed(start)
        results.append(Result(seconds, mse(train, centroids), mse(held_out, centroids)))
    return results


def median_over_seeds(runs: list[list[Result]]) -> list[Result]:
    return [Result(*map(statistics.median, zip(*per_budget))) for per_budget in zip(*runs)]


def print_table(lloyd_results: list[Result], e_means_results: list[Result]) -> None:
    reference = lloyd_results[-1].held_out_mse
    rows = [("Lloyd", i, lloyd_results[i - 1]) for i in LLOYD_REPORTED_ITERATIONS]
    rows += [("e-means", epochs, result) for epochs, result in zip(E_MEANS_EPOCHS, e_means_results)]
    print(f"\n{'method':<8} {'epochs':>6} {'seconds':>8} {'train MSE':>10} {'held-out MSE':>13} {'vs Lloyd@25':>12}")
    for name, epochs, result in rows:
        change = 100 * (result.held_out_mse / reference - 1)
        print(
            f"{name:<8} {epochs:>6} {result.seconds:>8.2f} {result.train_mse:>10.0f}"
            f" {result.held_out_mse:>13.0f} {change:>+11.2f}%"
        )


def plot(lloyd_results: list[Result], e_means_results: list[Result], title: str, path: Path) -> None:
    figure, (by_epoch, by_time) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = (
        (by_epoch, range(1, LLOYD_ITERATIONS + 1), E_MEANS_EPOCHS, "epochs (passes over the data)"),
        (by_time, [r.seconds for r in lloyd_results], [r.seconds for r in e_means_results], "GPU seconds"),
    )
    reference = lloyd_results[-1].held_out_mse
    for axis, lloyd_x, e_means_x, x_label in panels:
        axis.plot(lloyd_x, [r.held_out_mse for r in lloyd_results], color=LLOYD_COLOR,
                  linewidth=2, marker="o", markersize=4, label="Lloyd's k-means (per iteration)")
        axis.plot(e_means_x, [r.held_out_mse for r in e_means_results], color=E_MEANS_COLOR,
                  linewidth=2, marker="o", markersize=8, label="e-means (one run per epoch budget)")
        axis.axhline(reference, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
        axis.annotate("Lloyd @ 25 iterations", (1, reference), xycoords=("axes fraction", "data"),
                      xytext=(-4, -4), textcoords="offset points", ha="right", va="top", color="#52514e", fontsize=9)
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
        axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.set_xlabel(x_label)
        axis.grid(color="#e4e3df", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    # Lloyd's first iterations sit far above the region where the two methods compete.
    by_epoch.set_ylim(top=max(lloyd_results[2].held_out_mse, e_means_results[0].held_out_mse) * 1.005)
    by_epoch.set_ylabel("held-out MSE (median over seeds)")
    by_epoch.legend(frameon=False)
    figure.suptitle(title)
    figure.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2048, help="number of clusters (default: n/500, as in the paper)")
    parser.add_argument("--seeds", type=int, default=5, help="runs per method, each from a new random init")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This comparison needs a CUDA GPU.")

    base, learn = load_sift1m(args.data_dir)
    train, held_out = torch.from_numpy(base).cuda(), torch.from_numpy(learn).cuda()
    n, d = train.shape
    print(f"SIFT1M: {n} training vectors, {len(held_out)} held-out, d={d}, k={args.k}")
    for epochs in E_MEANS_EPOCHS:
        batch_size, alpha0 = e_means_recipe(n, args.k, d, epochs)
        print(f"  e-means recipe for {epochs} epochs: batch {batch_size}, alpha0 {alpha0:.3f}")
    e_means(train, train[: args.k], 1, seed=0)  # warm up the CUDA kernels both methods use, untimed

    lloyd_runs, e_means_runs = [], []
    for seed in range(args.seeds):
        init = train[np.random.default_rng(seed).choice(n, args.k, replace=False)]
        lloyd_runs.append(run_lloyd(train, held_out, init))
        e_means_runs.append(run_e_means(train, held_out, init, seed))
        print(f"  seed {seed} done")

    lloyd_results, e_means_results = median_over_seeds(lloyd_runs), median_over_seeds(e_means_runs)
    print_table(lloyd_results, e_means_results)
    path = Path("results.png")
    plot(lloyd_results, e_means_results, f"SIFT1M, k={args.k}, median of {args.seeds} seeds", path)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    main()
