"""Regularized kernel Stein estimator of the local Fisher score.

Appendix E.1 specifies the kernel, the median bandwidth, and the ridge.
Equation (11) is the sample score. Derivatives act on the target coordinate
of the first kernel argument. The score is Ji(T) = E[r - s^2], which equals
minus the local Fisher information, not a conditional variance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

# Selected synthetic configuration in Appendix E.3.
BANDWIDTH_MULTIPLIER = 1.5
RIDGE_MULTIPLIER = 0.5
RIDGE_FLOOR = 1e-10


def stein_scores(Z, u=BANDWIDTH_MULTIPLIER, r=RIDGE_MULTIPLIER):
    """Local Fisher scores of every coordinate of Z, shape (n, k) -> (k,).

    Columns are mean-centered here. Variance is not standardized.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("float64 is required; enable jax_enable_x64")
    Z = jnp.asarray(Z, dtype=jnp.float64)
    if Z.ndim != 2:
        raise ValueError("Z must have shape (n, k)")
    n, k = Z.shape
    if n < 2 or k < 1:
        raise ValueError("need at least two rows and one column")
    Z = Z - jnp.mean(Z, axis=0, keepdims=True)

    sq = jnp.sum(Z * Z, axis=1)
    gram = Z @ Z.T
    d2 = jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    # Median of all n^2 pairwise Euclidean distances, diagonal included.
    bandwidth = float(u) * jnp.median(jnp.sqrt(d2))
    bandwidth = float(bandwidth)
    if not jnp.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("nonpositive bandwidth")

    inv_h = 1.0 / bandwidth
    inv_h2 = inv_h * inv_h
    kernel = inv_h * jnp.exp(-0.5 * d2 * inv_h2)
    eta = max(RIDGE_FLOOR, float(r) * float(jnp.trace(kernel)) / (n * k))
    system = kernel + eta * jnp.eye(n, dtype=jnp.float64)
    try:
        factor = cho_factor(system, lower=True, check_finite=True)
    except Exception as exc:
        raise RuntimeError("failed factorization") from exc

    ones = jnp.ones(n, dtype=jnp.float64)
    kernel_ones = kernel @ ones
    kernel_Z = kernel @ Z
    # D1_b = sum_a d/d z_a,t k(z_a, z_b) = -h^{-2} ((K Z)_b - z_b (K 1)_b)
    d1 = -inv_h2 * (kernel_Z - Z * kernel_ones[:, None])
    z2 = Z * Z
    sum_delta2 = (kernel @ z2) - 2.0 * Z * kernel_Z + z2 * kernel_ones[:, None]
    d2_target = sum_delta2 * (inv_h2 * inv_h2) - inv_h2 * kernel_ones[:, None]
    score = cho_solve(factor, -d1, check_finite=True)
    curvature = cho_solve(factor, d2_target, check_finite=True)
    if not bool(jnp.all(jnp.isfinite(score))) or not bool(jnp.all(jnp.isfinite(curvature))):
        raise RuntimeError("failed factorization")
    # Equation (11): (1/n) 1^T r_hat - (1/n) ||s_hat||^2, with sample values.
    local = jnp.mean(curvature, axis=0) - jnp.mean(score * score, axis=0)
    return local


def local_fisher_score(X, target, conditioning=(), u=BANDWIDTH_MULTIPLIER, r=RIDGE_MULTIPLIER):
    """Estimated local Fisher score of one node given a conditioning set.

    X has shape (n, p). `target` is a column index. `conditioning` is an
    iterable of other column indices. The returned value estimates Ji(T).
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    target = int(target)
    cond = tuple(int(j) for j in conditioning)
    if target < 0 or target >= X.shape[1] or any(j < 0 or j >= X.shape[1] for j in cond):
        raise ValueError("column index out of range")
    columns = (target,) + cond
    values = stein_scores(X[:, columns], u=u, r=r)
    return float(values[0])
