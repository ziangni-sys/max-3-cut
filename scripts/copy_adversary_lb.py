"""Lower bound on alpha_i = C_i/(C_i - S2_i) valid for EVERY i-gadget (any auxiliary columns,
any size), via LP duality with a restricted family of adversaries.

Relaxed primal (per predicate i, NRT framework; sat-class soundness rows dropped, which can only
increase the optimum):  max g  s.t.  cut(w; canonical at x) >= 1  (x in C_i),
                                      cut(w; any coloring with x unsat, y in Y) + g <= 1,  w >= 0,
over the universe of all auxiliary columns f: C_i -> Z_k.  Weak duality: for any
pi >= 0 on C_i and any probability distribution nu over unsat "rows" such that every possible
edge e satisfies  E_nu[cut_e] >= sum_x pi_x cut_e(canonical x),  we get g <= 1 - sum(pi).

Copy adversary: a row is (x unsat, y in Y, phi in C_i, sigma in S_k); every auxiliary vertex f
is coloured sigma(f(phi)).  Then the constraints over all (infinitely many) edges reduce to:
  aux-aux:    nu(phi = x') >= pi_x'                                   for all x' in C_i
  aux-fixed t: sum_{x'} min_v [ sum_{rows: phi=x'} nu [sigma(v) != col_row(t)]
                                - pi_x' [v != canon_t(x')] ] >= 0
  fixed-fixed: E_nu[col(t) != col(t')] >= sum_x' pi_x' [canon_t(x') != canon_t'(x')]
Maximize sum(pi);  alpha_i >= 1/(1 - sum(pi)).
"""
import sys
import os
from itertools import permutations, product
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from gadgets.core import allowed_global_assignments, sat_assignments


def copy_adversary_bound(k, i, verbose=True):
    Y = allowed_global_assignments(k)
    sat = sat_assignments(k, i)
    S = len(sat)
    uns = [x for x in product(range(k), repeat=3) if sum(x) % k != i]
    sig = list(permutations(range(k)))
    rows = [(x, y, p, s) for x in uns for y in Y for p in range(S) for s in range(len(sig))]
    R = len(rows)
    fixed = [('p', j) for j in range(3)] + [('g', c) for c in range(k)]
    T = len(fixed)

    def col_row(r, t):
        x, y, p, s = rows[r]
        return x[t[1]] if t[0] == 'p' else y[t[1]]

    def canon(t, xp):
        return xp[t[1]] if t[0] == 'p' else t[1]

    # variable layout: pi (S) | nu (R) | m (S*T)
    nP, nN, nM = S, R, S * T
    off_nu, off_m = nP, nP + nN
    nv = nP + nN + nM
    A_ub, b_ub = [], []
    rr, cc, vv = [], [], []
    nrow = 0

    def add(entries, b):
        nonlocal nrow
        for j, v in entries:
            rr.append(nrow); cc.append(j); vv.append(v)
        b_ub.append(b)
        nrow += 1

    phi_rows = [[] for _ in range(S)]
    for r, (x, y, p, s) in enumerate(rows):
        phi_rows[p].append(r)
    # aux-aux:  pi_x' - nu(phi=x') <= 0
    for p in range(S):
        add([(p, 1.0)] + [(off_nu + r, -1.0) for r in phi_rows[p]], 0.0)
    # aux-fixed: m_{x',t} - sum_{phi=x'} nu [sigma(v) != col(t)] + pi_x'[v != canon] <= 0 ; -sum_x' m <= 0
    for ti, t in enumerate(fixed):
        for p in range(S):
            for v in range(k):
                ent = [(off_m + p * T + ti, 1.0)]
                for r in phi_rows[p]:
                    x, y, _, s = rows[r]
                    if sig[s][v] != col_row(r, t):
                        ent.append((off_nu + r, -1.0))
                if v != canon(t, sat[p]):
                    ent.append((p, 1.0))
                add(ent, 0.0)
        add([(off_m + p * T + ti, -1.0) for p in range(S)], 0.0)
    # fixed-fixed: sum_x' pi [canon differ] - E_nu[col differ] <= 0
    for a in range(T):
        for b in range(a + 1, T):
            ent = []
            for p in range(S):
                if canon(fixed[a], sat[p]) != canon(fixed[b], sat[p]):
                    ent.append((p, 1.0))
            for r in range(R):
                if col_row(r, fixed[a]) != col_row(r, fixed[b]):
                    ent.append((off_nu + r, -1.0))
            add(ent, 0.0)
    A = sp.csr_matrix((vv, (rr, cc)), shape=(nrow, nv))
    A_eq = sp.csr_matrix(([1.0] * R, ([0] * R, list(range(off_nu, off_nu + R)))), shape=(1, nv))
    c = np.zeros(nv); c[:nP] = -1.0
    bounds = [(0, None)] * (nP + nN) + [(None, None)] * nM
    res = linprog(c, A_ub=A, b_ub=np.array(b_ub), A_eq=A_eq, b_eq=[1.0], bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(res.message)
    spi = -res.fun
    lb = 1.0 / (1.0 - spi) if spi < 1 else float('inf')
    if verbose:
        print(f'k={k} i={i}: rows={R} sum(pi)={spi:.6f}  ->  alpha >= {lb:.4f}', flush=True)
    return lb, res


if __name__ == '__main__':
    for k in [int(a) for a in sys.argv[1].split(',')]:
        for i in range(k):
            copy_adversary_bound(k, i)
