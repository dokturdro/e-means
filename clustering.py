"""Lloyd's k-means and e-means on PyTorch tensors.

e-means follows Algorithm 1 of Bessa et al., "e-means: outperforming k-means (in one-tenth the
epochs)", arXiv:2611.09501, with the scaling-law constants shipped in faiss/contrib/e_means/base.py.
"""

import math
from collections.abc import Iterator

import torch

# Rows scored at once in `assign`: bounds the (rows, k) distance matrix in GPU memory.
ASSIGN_CHUNK_SIZE = 65536

# Paper Algorithm 1: initial EMA mass N_j, revival every 5 batches below tau * mean(N).
INITIAL_MASS = 1e-10
REVIVE_EVERY = 5
REVIVAL_THRESHOLD = 0.1
# Equation 5: the cosine schedule decays the step from alpha0 to MIN_STEP_FACTOR * alpha0.
MIN_STEP_FACTOR = 0.1
# Relative perturbation separating a split donor from its child (paper Appendix J, as in faiss).
SPLIT_EPSILON = 1e-4


def assign(x: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared L2 distance to, and index of, each row's nearest centroid."""
    centroid_norms = centroids.square().sum(1)
    distances, labels = [], []
    for chunk in x.split(ASSIGN_CHUNK_SIZE):
        partial = torch.addmm(centroid_norms, chunk, centroids.T, alpha=-2)  # ||c||^2 - 2 x.c
        best, label = partial.min(1)
        distances.append((best + chunk.square().sum(1)).clamp_(min=0))
        labels.append(label)
    return torch.cat(distances), torch.cat(labels)


def mse(x: torch.Tensor, centroids: torch.Tensor) -> float:
    """Mean squared distance of the rows of x to their nearest centroid."""
    return assign(x, centroids)[0].mean().item()


def cluster_sums(x: torch.Tensor, labels: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cluster sum of the assigned rows and number of assigned rows."""
    sums = torch.zeros(k, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, labels, x)
    counts = torch.bincount(labels, minlength=k).to(x.dtype)
    return sums, counts


def lloyd(x: torch.Tensor, init: torch.Tensor, iterations: int) -> Iterator[torch.Tensor]:
    """Yield the centroids after each Lloyd iteration. Empty clusters keep their centroid."""
    centroids = init
    for _ in range(iterations):
        _, labels = assign(x, centroids)
        sums, counts = cluster_sums(x, labels, len(centroids))
        means = sums / counts.clamp(min=1)[:, None]
        centroids = torch.where((counts > 0)[:, None], means, centroids)
        yield centroids


def round_to_power_of_two(value: float) -> int:
    return max(1, 2 ** round(math.log2(value)))


def e_means_recipe(n: int, k: int, d: int, epochs: int) -> tuple[int, float]:
    """Batch size (Equation 7) and initial step alpha0 (Equation 6) from the fitted scaling laws.

    At d <= 16 the library defaults to its "quality" law, with batches 8x smaller than the
    "throughput" law, because more updates pay off most at low dimension.
    """
    coefficient = 0.00439175 if d <= 16 else 0.035134
    batch = coefficient * k**0.134749 * n**0.876473 * epochs**0.813344
    batch_size = min(round_to_power_of_two(batch), n // 2)
    alpha0 = 0.8244 * k**-0.4617 * batch_size**0.9003 * n**-0.4498 * epochs**-0.1057
    return batch_size, min(max(alpha0, 1e-4), 1.0)


def scheduled_step(alpha0: float, step: int, total_steps: int) -> float:
    """Equation 5: cosine decay of the EMA step from alpha0 to MIN_STEP_FACTOR * alpha0."""
    cosine = 0.5 * (1 + math.cos(math.pi * step / total_steps))
    return alpha0 * (MIN_STEP_FACTOR + (1 - MIN_STEP_FACTOR) * cosine)


def ema_update(
    centroids: torch.Tensor, mass: torch.Tensor, sums: torch.Tensor, counts: torch.Tensor, alpha: float
) -> None:
    """Equation 2, S <- (1-a) S + a s and N <- (1-a) N + a n, applied in place.

    The state is stored as the centroid c = S / N and the mass N, an exact rewrite that avoids
    S and N underflowing to 0 / 0 for clusters that stop receiving points.
    """
    mass *= 1 - alpha
    hit = counts > 0
    new_mass = mass[hit] + alpha * counts[hit]
    batch_means = sums[hit] / counts[hit, None]
    weight = (alpha * counts[hit] / new_mass)[:, None]
    centroids[hit] += weight * (batch_means - centroids[hit])
    mass[hit] = new_mass


def revive_dead_clusters(centroids: torch.Tensor, mass: torch.Tensor, generator: torch.Generator) -> None:
    """Re-seed clusters whose mass fell below REVIVAL_THRESHOLD * mean mass by splitting busy ones.

    Donors are drawn without replacement in proportion to their mass. Each donor shares its mass
    with the revived cluster, and the two are nudged apart along alternating coordinate signs.
    """
    dead = mass < REVIVAL_THRESHOLD * mass.mean()
    candidates = ~dead & (mass > 0)
    split_count = min(int(dead.sum()), int(candidates.sum()))
    if split_count == 0:
        return
    donors = torch.multinomial(mass * candidates, split_count, replacement=False, generator=generator)
    targets = dead.nonzero().squeeze(1)[:split_count]
    sign = torch.ones(centroids.shape[1], device=centroids.device)
    sign[1::2] = -1
    donor_centroids = centroids[donors]
    centroids[targets] = donor_centroids * (1 + SPLIT_EPSILON * sign)
    centroids[donors] = donor_centroids * (1 - SPLIT_EPSILON * sign)
    mass[donors] /= 2
    mass[targets] = mass[donors]


def e_means(x: torch.Tensor, init: torch.Tensor, epochs: int, seed: int) -> torch.Tensor:
    """Algorithm 1: mini-batch EMA k-means, closed by the exact means of the last epoch's assignments."""
    n, d = x.shape
    k = len(init)
    batch_size, alpha0 = e_means_recipe(n, k, d, epochs)
    total_steps = epochs * math.ceil(n / batch_size)
    generator = torch.Generator(x.device).manual_seed(seed)
    centroids = init.clone()
    mass = torch.full((k,), INITIAL_MASS, device=x.device)
    step = 0
    for epoch in range(epochs):
        # Algorithm 1, line 12: the last epoch's assignments are kept for the closing means.
        epoch_sums = torch.zeros_like(centroids)
        epoch_counts = torch.zeros(k, device=x.device)
        for batch_ids in torch.randperm(n, device=x.device, generator=generator).split(batch_size):
            batch = x[batch_ids]
            _, labels = assign(batch, centroids)
            sums, counts = cluster_sums(batch, labels, k)
            if epoch == epochs - 1:
                epoch_sums += sums
                epoch_counts += counts
            ema_update(centroids, mass, sums, counts, scheduled_step(alpha0, step, total_steps))
            step += 1
            if step % REVIVE_EVERY == 0 and d > 2:  # the library disables revival at d <= 2
                revive_dead_clusters(centroids, mass, generator)
    assigned = epoch_counts > 0
    centroids[assigned] = epoch_sums[assigned] / epoch_counts[assigned, None]
    return centroids
