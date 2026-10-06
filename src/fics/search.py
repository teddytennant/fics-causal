"""Source-first Fisher Information Completion Search (Algorithm 1)."""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple

import jax.numpy as jnp

from fics.score import BANDWIDTH_MULTIPLIER, RIDGE_MULTIPLIER, stein_scores

# Selected synthetic parent-penalty multiplier in Appendix E.3.
PENALTY_MULTIPLIER = 0.8


class SearchResult(NamedTuple):
    order: tuple
    parents: dict
    edges: tuple


def search(X, q, u=BANDWIDTH_MULTIPLIER, r=RIDGE_MULTIPLIER, c=PENALTY_MULTIPLIER):
    """Run Algorithm 1 and return a DAG on columns of X.

    Ordering minimizes the local Fisher score over conditioning sets of size
    min(q, |S|) drawn from already ordered nodes. A source, whose empty-set
    score equals its noise Fisher, is therefore ordered first.

    The parent set of a node is the minimizer of Ji(T) + lambda |T| over
    predecessor subsets of size at most q. Lambda is c log(p) / n, natural
    log. Ties prefer fewer parents, then lexicographic order. Edges run only
    from earlier nodes to later nodes, so the output is acyclic.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    n, p = (int(X.shape[0]), int(X.shape[1]))
    if n < 2 or p < 1:
        raise ValueError("need at least two observations and one node")
    q = int(q)
    if q < 0:
        raise ValueError("q must be nonnegative")

    cache = {}

    def score(target, cond):
        nodes = frozenset(cond) | {int(target)}
        stored = cache.get(nodes)
        if stored is None:
            idx = tuple(sorted(nodes))
            values = stein_scores(X[:, idx], u=u, r=r)
            stored = {idx[t]: float(values[t]) for t in range(len(idx))}
            cache[nodes] = stored
        return stored[int(target)]

    # log(1) = 0. Natural log, as in the log(p)/n penalty.
    lam = float(c) * math.log(p) / float(n) if p > 1 else 0.0
    selected = []
    selected_set = set()
    while len(selected) < p:
        q_s = min(q, len(selected))
        best_i = None
        best_score = math.inf
        for i in range(p):
            if i in selected_set:
                continue
            if q_s == 0:
                score_i = score(i, ())
            else:
                score_i = math.inf
                for subset in itertools.combinations(selected, q_s):
                    score_i = min(score_i, score(i, subset))
            if best_i is None or score_i < best_score or (score_i == best_score and i < best_i):
                best_score = score_i
                best_i = i
        selected.append(best_i)
        selected_set.add(best_i)

    parents = {}
    edges = []
    for pos, i in enumerate(selected):
        preds = selected[:pos]
        best = ()
        best_key = None
        for size in range(0, min(q, len(preds)) + 1):
            for subset in itertools.combinations(tuple(sorted(preds)), size):
                objective = score(i, subset) + lam * size
                key = (objective, size, subset)
                if best_key is None or key < best_key:
                    best_key = key
                    best = subset
        parents[i] = best
        for parent in best:
            edges.append((parent, i))
    edges = tuple(sorted(edges))
    return SearchResult(order=tuple(selected), parents=parents, edges=edges)
