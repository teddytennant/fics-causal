"""Seeded CPU checks for the local Fisher score and source-first search."""

import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from fics import local_fisher_score, search


def _is_dag(edges, p):
    indegree = [0] * p
    children = [[] for _ in range(p)]
    for parent, child in edges:
        children[parent].append(child)
        indegree[child] += 1
    queue = [i for i in range(p) if indegree[i] == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return seen == p


def _topo_order(order, edges):
    position = {node: i for i, node in enumerate(order)}
    return all(position[parent] < position[child] for parent, child in edges)


def _chain(n, p, seed, coef, noise):
    rng = np.random.default_rng(seed)
    data = np.zeros((n, p), dtype=np.float64)
    data[:, 0] = rng.normal(0.0, noise, size=n)
    for i in range(1, p):
        data[:, i] = coef * data[:, i - 1] + rng.normal(0.0, noise, size=n)
    return data


def _two_parent_dag(n, seed, noise):
    """Linear Gaussian ANM, indegree 2. Edges are not known to the search."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, noise, size=n)
    x1 = rng.normal(0.0, noise, size=n)
    x2 = 1.2 * x0 + 0.9 * x1 + rng.normal(0.0, noise, size=n)
    x3 = 1.1 * x1 + 1.0 * x2 + rng.normal(0.0, noise, size=n)
    x4 = 0.8 * x2 + 1.3 * x3 + rng.normal(0.0, noise, size=n)
    return np.column_stack([x0, x1, x2, x3, x4]).astype(np.float64)


TRUE_DAG_EDGES = ((0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4))


def test_score_gap_matches_noise_fisher():
    # X -> Y, nonconstant parent. Noise Fisher information is 1/sigma^2.
    n = 1500
    noise = 5.0
    coef = 1.2
    sigma2 = noise * noise
    noise_score = -1.0 / sigma2
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, noise, size=n)
    y = coef * x + rng.normal(0.0, noise, size=n)
    data = np.column_stack([x, y]).astype(np.float64)

    score_empty = local_fisher_score(data, target=1, conditioning=())
    score_parent = local_fisher_score(data, target=1, conditioning=(0,))
    score_source = local_fisher_score(data, target=0, conditioning=())

    assert score_empty > score_parent
    assert abs(score_parent - noise_score) < abs(score_empty - noise_score)
    assert abs(score_parent - noise_score) < 0.005
    assert abs(score_source - noise_score) < 0.005
    assert score_empty - score_parent > 0.01


def test_search_recovers_exact_dag():
    n = 1500
    noise = 5.0
    recovered = []
    for seed in (0, 1):
        data = _two_parent_dag(n, seed, noise)
        result = search(data, q=2)
        assert len(result.order) == 5
        assert set(result.order) == set(range(5))
        assert _is_dag(result.edges, 5)
        assert _topo_order(result.order, result.edges)
        assert all(len(result.parents[i]) <= 2 for i in range(5))
        recovered.append(set(result.edges) == set(TRUE_DAG_EDGES))
    assert any(recovered)


def test_incomplete_fit_is_still_a_dag():
    # Unit noise makes the Stein gap between a parent set and a redundant
    # superset larger than lambda, so exact edges are not recovered. The
    # predecessor restriction still returns a DAG.
    data = _chain(n=1500, p=5, seed=0, coef=1.0, noise=1.0)
    true_edges = {(i, i + 1) for i in range(4)}
    result = search(data, q=2)
    assert _is_dag(result.edges, 5)
    assert _topo_order(result.order, result.edges)
    assert set(result.edges) != true_edges
