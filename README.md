FiCS is a small JAX implementation of Fisher Information Completion Search from arXiv:2610.04291.
pip install -e .
python3 -m pytest
This is the source-first search and the regularized kernel Stein local Fisher score on synthetic additive noise models. It does not run the paper's real-data benchmarks. Exact recovery is checked at noise standard deviation 5, not unit Gaussian noise, because at unit noise and n=1500 the Stein gap exceeds the parent penalty.
