"""Fisher Information Completion Search (arXiv:2610.04291)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from fics.score import local_fisher_score
from fics.search import search

__all__ = ["local_fisher_score", "search"]
