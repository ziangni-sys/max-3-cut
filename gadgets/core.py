"""Exact evaluation of 3LIN(k) -> MAX-k-CUT gadgets in the framework of
Nagda-Raghavan-Thakurta (arXiv:2509.18057, Appendix A, Def. A.1-A.2, Thm A.4).

Conventions (0-indexed internally):
  vertices 0,1,2          primary variables x
  vertices 3 .. 3+k-1     global variables y
  vertices 3+k .. n-1     auxiliary variables z
A gadget is a symmetric nonnegative integer matrix W (n x n, zero diagonal);
its value on a coloring col in Z_k^n is sum_{u<v} W[u,v] * [col_u != col_v].

For the i-gadget (source predicate x1+x2+x3 = i mod k):
  C  = min_{x in C_i} max_z I(x, y0, z),  y0 = (0,1,...,k-1)
  S1 = max_{x in C_i, y in Y, z} I(x, y, z)
  S2 = max_{x in S_i, y in Y, z} I(x, y, z)
  t  = total weight
where Y = global assignments that are lexicographically minimal under
permutations of the alphabet (restricted growth strings).
Thm A.4: hardness ratio  b/a = [sum_i S2_i + max_i (S1_i - S2_i)] / sum_i C_i.
"""
from fractions import Fraction
from itertools import product
import numpy as np


def allowed_global_assignments(k):
    """Y: restricted growth strings of length k (as in NRT's code snippet)."""
    ret = []
    for y in product(range(k), repeat=k):
        seen = []
        for e in y:
            if e not in seen:
                seen.append(e)
        if seen == list(range(len(seen))):
            ret.append(y)
    return ret


def sat_assignments(k, i):
    return [x for x in product(range(k), repeat=3) if sum(x) % k == i]


def unsat_assignments(k, i):
    return [x for x in product(range(k), repeat=3) if sum(x) % k != i]


def edges_to_matrix(edges, n=None, one_indexed=True):
    off = 1 if one_indexed else 0
    if n is None:
        n = max(max(a, b) for a, b, _ in edges) + 1 - off
    W = np.zeros((n, n), dtype=np.int64)
    for a, b, w in edges:
        a -= off
        b -= off
        assert a != b
        W[a, b] += w
        W[b, a] += w
    return W


def relabel_globals(W, k, perm):
    """Return the gadget in which global vertex 3+j is replaced by 3+perm[j]
    (i.e. the edge set is the same but the global variables are renamed)."""
    n = W.shape[0]
    p = list(range(n))
    for j in range(k):
        p[3 + j] = 3 + perm[j]
    W2 = np.zeros_like(W)
    for u in range(n):
        for v in range(n):
            W2[p[u], p[v]] = W[u, v]
    return W2


# ---------------------------------------------------------------------------
# exact max over the auxiliary coloring, with the other vertices fixed
# ---------------------------------------------------------------------------

def _enum(k, m):
    if m == 0:
        return np.zeros((1, 0), dtype=np.int64)
    return np.array(list(product(range(k), repeat=m)), dtype=np.int64)


class FixedMaxCut:
    """Exact max_z of the gadget value with primaries+globals fixed.

    Meet-in-the-middle over a split A|B of the auxiliary vertices: the value is
    f_A(z_A) + f_B(z_B) + cross(z_A, z_B) with cross = totAB - sum_c E_Ac W_AB E_Bc^T.
    Exact for integer weights (float64 sums of integers < 2^53).
    """

    def __init__(self, W, k, aux=None, integer=True):
        self.integer = integer
        self.W = np.asarray(W, dtype=np.int64 if integer else np.float64)
        self.k = k
        n = self.W.shape[0]
        self.fixed = list(range(3 + k))
        self.aux = list(range(3 + k, n)) if aux is None else list(aux)
        m = len(self.aux)
        self.A = self.aux[: (m + 1) // 2]
        self.B = self.aux[(m + 1) // 2:]
        self.ZA = _enum(k, len(self.A))
        self.ZB = _enum(k, len(self.B))
        W = self.W
        # pairwise terms inside A and inside B (independent of the fixed colors)
        self.pairA = self._pair_within(self.A, self.ZA)
        self.pairB = self._pair_within(self.B, self.ZB)
        WAB = W[np.ix_(self.A, self.B)].astype(np.float64)
        self.totAB = float(WAB.sum())
        # E_c: indicator matrices
        self.EA = [(self.ZA == c).astype(np.float64) for c in range(k)]
        self.EB = [(self.ZB == c).astype(np.float64) for c in range(k)]
        if len(self.A) and len(self.B):
            eq = np.zeros((len(self.ZA), len(self.ZB)))
            for c in range(k):
                eq += (self.EA[c] @ WAB) @ self.EB[c].T
            self.cross = self.totAB - eq
        else:
            self.cross = np.zeros((len(self.ZA), len(self.ZB)))

    def _pair_within(self, S, Z):
        W = self.W
        val = np.zeros(len(Z), dtype=W.dtype)
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                w = W[S[a], S[b]]
                if w:
                    val += w * (Z[:, a] != Z[:, b])
        return val

    def _unary(self, S, Z, col_fixed):
        # sum_{u in S} sum_{v fixed} W[u,v] [z_u != col_v]
        W = self.W
        val = np.zeros(len(Z), dtype=W.dtype)
        for a, u in enumerate(S):
            # gain[c] for vertex u taking color c
            gain = np.zeros(self.k, dtype=W.dtype)
            for v, cv in zip(self.fixed, col_fixed):
                w = W[u, v]
                if w:
                    gain += w
                    gain[cv] -= w
            val += gain[Z[:, a]]
        return val

    def fixed_value(self, col_fixed):
        W = self.W
        s = 0
        F = self.fixed
        for a in range(len(F)):
            for b in range(a + 1, len(F)):
                if col_fixed[a] != col_fixed[b]:
                    s += W[F[a], F[b]]
        return s

    def maxval(self, x, y, return_arg=False):
        col_fixed = list(x) + list(y)
        fA = (self.pairA + self._unary(self.A, self.ZA, col_fixed)).astype(np.float64)
        fB = (self.pairB + self._unary(self.B, self.ZB, col_fixed)).astype(np.float64)
        tot = fA[:, None] + fB[None, :] + self.cross
        idx = int(np.argmax(tot))
        if self.integer:
            best = int(round(tot.flat[idx])) + int(self.fixed_value(col_fixed))
        else:
            best = float(tot.flat[idx]) + float(self.fixed_value(col_fixed))
        if not return_arg:
            return best
        ia, ib = divmod(idx, len(self.ZB))
        z = {}
        for a, u in enumerate(self.A):
            z[u] = int(self.ZA[ia, a])
        for b, u in enumerate(self.B):
            z[u] = int(self.ZB[ib, b])
        return best, z


def value(W, col):
    col = np.asarray(col)
    diff = col[:, None] != col[None, :]
    return int((W * diff).sum() // 2)


def gadget_params(W, k, i, solver=None, verbose=False):
    """Return dict with C, S1, S2, t for the i-gadget W (exact)."""
    if solver is None:
        solver = FixedMaxCut(W, k)
    Y = allowed_global_assignments(k)
    y0 = tuple(range(k))
    sat = sat_assignments(k, i)
    uns = unsat_assignments(k, i)
    C = min(solver.maxval(x, y0) for x in sat)
    S1 = max(solver.maxval(x, y) for x in sat for y in Y)
    S2 = max(solver.maxval(x, y) for x in uns for y in Y)
    t = int(np.triu(W, 1).sum())
    return dict(C=C, S1=S1, S2=S2, t=t)


def hardness_ratio(params):
    """Thm A.4 of NRT: b/a for a list of per-i parameter dicts (exact Fraction)."""
    sumC = sum(p['C'] for p in params)
    sumS2 = sum(p['S2'] for p in params)
    gap = max(p['S1'] - p['S2'] for p in params)
    return Fraction(sumS2 + gap, sumC)


def gadget_params_cpsat(W, k, i, workers=1, time_limit=600.0):
    """Same as gadget_params but every inner max is solved (and proved optimal) by CP-SAT."""
    from .cpsat import cpsat_max
    W = np.asarray(W, dtype=np.int64)
    nf = 3 + k
    Y = allowed_global_assignments(k)
    y0 = tuple(range(k))
    cache = {}

    def mx(x, y):
        key = (x, y)
        if key not in cache:
            cache[key] = cpsat_max(W, k, list(x) + list(y), nf, time_limit=time_limit, workers=workers)[0]
        return cache[key]
    C = min(mx(x, y0) for x in sat_assignments(k, i))
    S1 = max(mx(x, y) for x in sat_assignments(k, i) for y in Y)
    S2 = max(mx(x, y) for x in unsat_assignments(k, i) for y in Y)
    t = int(np.triu(W, 1).sum())
    return dict(C=C, S1=S1, S2=S2, t=t)
