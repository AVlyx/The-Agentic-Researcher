"""Core library for multipartite weak Schur sampling (MWS).

Notation
--------
n parties, local dimensions d = (d_1, ..., d_n), H_i = C^{d_i},
H = H_1 (x) ... (x) H_n.  For a pure state |psi> in H we take k i.i.d. copies,
|psi^{(x)k}> in H^{(x)k}.  Regrouping copies by party,

    H^{(x)k}  ~=  (x)_{i=1}^n  H_i^{(x)k},

and for a subset (block) B subset [n] with H_B = (x)_{i in B} H_i we let
R_B(sigma), sigma in S_k, be the representation of S_k permuting the k copies of
H_B, i.e. the simultaneous permutation of the copy-axes of every party i in B.

    P^{(B)}_lambda = (f^lambda / k!) sum_{sigma in S_k} chi^lambda(sigma) R_B(sigma)

is the isotypic projector of Schur-Weyl duality on H_B^{(x)k}.

Tensor layout
-------------
A "copy tensor" T is a complex ndarray with n*k axes; axis (i*k + c) is party i,
copy c, of dimension d_i.  This layout makes R_B(sigma) a pure axis transpose.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from math import factorial

import numpy as np

from schur_weyl import dim_specht, dim_weyl, kronecker_coefficient, partitions
from schur_weyl.character import character
from schur_weyl.symmetric_group import permutation_cycle_type

# `kronecker` is the package implementation (schur_weyl >= 0.2.0); `kronecker_ref`
# below is an independent reimplementation kept as a cross-check.  They are
# verified to agree on all pairs/triples for k <= 5 in verify_support_theorem.py.
kronecker = kronecker_coefficient

# --------------------------------------------------------------------------
# symmetric group / characters / Kronecker coefficients
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _partitions(k: int, max_height: int | None = None) -> tuple[tuple[int, ...], ...]:
    return tuple(partitions(k, max_height=max_height))


@lru_cache(maxsize=None)
def class_size(mu: tuple[int, ...]) -> int:
    """|C_mu|, the size of the conjugacy class of cycle type mu in S_k."""
    k = sum(mu)
    mult: dict[int, int] = {}
    for part in mu:
        mult[part] = mult.get(part, 0) + 1
    denom = 1
    for part, m in mult.items():
        denom *= (part**m) * factorial(m)
    return factorial(k) // denom


@lru_cache(maxsize=None)
def kronecker_ref(*lams: tuple[int, ...]) -> int:
    """Independent reimplementation of the n-fold Kronecker coefficient g(lam^1,...,lam^n).

    Kept only as a cross-check on `schur_weyl.kronecker_coefficient`; use `kronecker`.

    Equals dim (S^{lam^1} (x) ... (x) S^{lam^n})^{S_k}, the multiplicity of the
    trivial representation in the tensor product of Specht modules:

        g = (1/k!) sum_{mu |- k} |C_mu| prod_i chi^{lam^i}(mu).

    Two arguments give the Kronecker delta; three give the classical Kronecker
    coefficient g(lam, mu, nu) (characters of S_k are real, so the
    trivial-multiplicity and Hom formulations agree).
    """
    if not lams:
        raise ValueError("need at least one partition")
    k = sum(lams[0])
    if any(sum(lam) != k for lam in lams):
        raise ValueError("all partitions must have the same size")
    total = 0
    for mu in _partitions(k):
        prod = class_size(mu)
        for lam in lams:
            c = character(lam, mu)
            if c == 0:
                prod = 0
                break
            prod *= c
        total += prod
    assert total % factorial(k) == 0, "Kronecker coefficient must be an integer"
    return total // factorial(k)


def mult_in_tensor(lams: tuple[tuple[int, ...], ...], nu: tuple[int, ...]) -> int:
    """Multiplicity of S^nu inside S^{lam^1} (x) ... (x) S^{lam^m}."""
    return kronecker(*lams, nu)


# --------------------------------------------------------------------------
# the S_k action on the copy tensor
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _perms(k: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.permutations(range(k)))


def permute_copies(
    T: np.ndarray,
    sigma: tuple[int, ...],
    block: tuple[int, ...],
    n: int,
    k: int,
    lead: int = 0,
) -> np.ndarray:
    """Apply R_B(sigma): permute the copy-axes of every party in `block`.

    `lead` leading axes of T are treated as an inert batch dimension.
    """
    axes = list(range(lead + n * k))
    for i in block:
        for c in range(k):
            axes[lead + i * k + c] = lead + i * k + sigma[c]
    return T.transpose(axes)


def apply_isotypic(
    T: np.ndarray,
    lam: tuple[int, ...],
    block: tuple[int, ...],
    n: int,
    k: int,
    lead: int = 0,
) -> np.ndarray:
    """Apply P^{(B)}_lambda to a copy tensor T, with B = `block`.

    Implemented as a weighted sum of axis transposes, so no d_B^k x d_B^k matrix
    is ever formed.  Cost: k! tensor-sized additions.  `lead` leading axes are a
    batch dimension.
    """
    out = np.zeros_like(T)
    for sigma in _perms(k):
        chi = character(lam, permutation_cycle_type(sigma))
        if chi == 0:
            continue
        out += chi * permute_copies(T, sigma, block, n, k, lead)
    return out * (dim_specht(lam) / factorial(k))


def sym_basis(dims: tuple[int, ...], k: int) -> np.ndarray:
    """Spanning set of Sym^k(H_1 (x) ... (x) H_n) in the (party, copy) layout.

    Returns an array of shape (m,) + (d_1,)*k + ... + (d_n,)*k with
    m = C(D + k - 1, k), D = prod d_i: the symmetrisations of the product basis.
    """
    n = len(dims)
    D = int(np.prod(dims))
    letters = list(itertools.product(*[range(d) for d in dims]))  # basis of H
    shape = tuple(dims[i] for i in range(n) for _ in range(k))
    vecs = []
    for combo in itertools.combinations_with_replacement(range(D), k):
        v = np.zeros(shape)
        for perm in set(itertools.permutations(combo)):
            idx = tuple(letters[perm[c]][i] for i in range(n) for c in range(k))
            v[idx] = 1.0
        vecs.append(v)
    return np.stack(vecs)


# --------------------------------------------------------------------------
# states and copy tensors
# --------------------------------------------------------------------------


def random_pure_state(dims: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure state on (x)_i C^{d_i}, returned with shape `dims`."""
    D = int(np.prod(dims))
    v = rng.normal(size=D) + 1j * rng.normal(size=D)
    v /= np.linalg.norm(v)
    return v.reshape(dims)


def copy_tensor(psi: np.ndarray, k: int) -> np.ndarray:
    """|psi^{(x)k}> in the (party, copy) layout: axis i*k + c is party i, copy c."""
    n = psi.ndim
    T = psi.astype(np.complex128)
    for _ in range(k - 1):
        T = np.tensordot(T, psi, axes=0)
    # current axis order is (copy, party); regroup to (party, copy)
    T = T.transpose([c * n + i for i in range(n) for c in range(k)])
    return np.ascontiguousarray(T)


def reduced_density_matrix(psi: np.ndarray, block: tuple[int, ...]) -> np.ndarray:
    """rho_B for a pure state psi given as an ndarray with one axis per party."""
    n = psi.ndim
    rest = tuple(i for i in range(n) if i not in block)
    M = psi.transpose(block + rest).reshape(
        int(np.prod([psi.shape[i] for i in block])), -1
    )
    return M @ M.conj().T


def schmidt_rank(psi: np.ndarray, block: tuple[int, ...], tol: float = 1e-9) -> int:
    """Schmidt rank of psi across the cut B | B^c."""
    ev = np.linalg.eigvalsh(reduced_density_matrix(psi, block))
    return int(np.sum(ev > tol))


# --------------------------------------------------------------------------
# the multipartite weak Schur sampling distribution
# --------------------------------------------------------------------------


def mws_distribution(
    psi: np.ndarray,
    k: int,
    blocks: tuple[tuple[int, ...], ...] | None = None,
    tol: float = 1e-12,
) -> dict[tuple[tuple[int, ...], ...], float]:
    """Joint outcome distribution of the measurement
    { prod_B P^{(B)}_{lambda^(B)} } over the family `blocks`.

    `blocks` defaults to the n singletons.  The blocks must form a laminar family
    (pairwise nested or disjoint) for the projectors to commute -- see
    verify_commutation.py.  Returns {tuple of partitions -> probability}, dropping
    outcomes below `tol`.
    """
    n = psi.ndim
    dims = psi.shape
    if blocks is None:
        blocks = tuple((i,) for i in range(n))
    dim_of = {B: int(np.prod([dims[i] for i in B])) for B in blocks}

    T0 = copy_tensor(psi, k)
    out: dict[tuple[tuple[int, ...], ...], float] = {}
    lam_choices = [_partitions(k, max_height=min(dim_of[B], k)) for B in blocks]
    for lams in itertools.product(*lam_choices):
        T = T0
        for B, lam in zip(blocks, lams):
            T = apply_isotypic(T, lam, B, n, k)
            if not np.any(np.abs(T) > 1e-14):
                break
        p = float(np.vdot(T, T).real)
        if p > tol:
            out[lams] = p
    return out


def support_prediction(
    dims: tuple[int, ...], k: int
) -> set[tuple[tuple[int, ...], ...]]:
    """Predicted support of the singleton-block measurement:
    { lam-tuples : g(lam^1,...,lam^n) > 0 and ell(lam^i) <= d_i }."""
    n = len(dims)
    lam_choices = [_partitions(k, max_height=min(dims[i], k)) for i in range(n)]
    return {lams for lams in itertools.product(*lam_choices) if kronecker(*lams) > 0}


def block_dimension(dims: tuple[int, ...], lams: tuple[tuple[int, ...], ...]) -> int:
    """dim( Im Pi_lam  cap  Sym^k(H) ) = g(lam^1,..,lam^n) * prod_i dim V^{d_i}_{lam^i}."""
    g = kronecker(*lams)
    if g == 0:
        return 0
    total = g
    for d, lam in zip(dims, lams):
        total *= dim_weyl(lam, d)
    return total
