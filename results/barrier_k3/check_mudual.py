"""Independent check of a fixed-mu dual certificate (mudual_t*.pkl): for EVERY pair of vertices of the
universe (3 primaries, 3 globals, all 3^9 columns on sat_t) verify
    sum_r nu_r [row r cuts e]  >=  sum_x pi_x [canonical colouring at x cuts e]   (up to tol),
that the rows are at y0 with unsatisfying x, and that the row weights have the prescribed marginal mu.
Then for every t-gadget  E_mu V_t(x, y0) >= sum(pi) * C_t.
Usage: check_mudual.py cert.pkl
"""
import sys
import os, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
sys.path.insert(0, _ROOT)
from itertools import product
import numpy as np
from gadgets.core import sat_assignments

d = pickle.load(open(sys.argv[1], 'rb'))
t, rows, nu, pi = d['t'], d['rows'], np.asarray(d['nu'], float), np.asarray(d['pi'], float)
K = 3
sat = sat_assignments(K, t)
S = len(sat)
DIG = np.array(list(product(range(K), repeat=S)), dtype=np.int64)
nF = len(DIG)
y0 = (0, 1, 2)
canfix = np.array([list(x) + list(y0) for x in sat])          # S x 6


def colour(row):
    if row[0] == 'L':
        _, T, h, x, y = row
        pw = K ** np.arange(len(T))
        pat = (DIG[:, list(T)] * pw[None, :]).sum(1)
        return np.asarray(h)[pat]
    _, c, sg, x, y = row
    return np.asarray(sg)[(DIG @ np.asarray(c)) % K]


keep = nu > 0
rows = [r for r, k in zip(rows, keep) if k]
nu = nu[keep]
R = len(rows)
assert all(tuple(r[4]) == y0 for r in rows), 'row not at y0'
assert all(sum(r[3]) % K != t for r in rows), 'row at a satisfying x'
COL = np.array([colour(r) for r in rows])                     # R x nF
FIX = np.array([list(r[3]) + list(r[4]) for r in rows])       # R x 6
# marginal over x-orbits
marg = {}
for r, w in zip(rows, nu):
    X = tuple(sorted(r[3]))
    marg[X] = marg.get(X, 0.0) + w
print('rows', R, 'sum nu', nu.sum(), 'sum pi', pi.sum())
print('marginal of nu on x-orbits:', {k: round(v, 9) for k, v in sorted(marg.items())})
print('prescribed mu:', dict(zip([tuple(i) for i in d['items']], np.round(d['mu_item'], 9))))
for X, m in zip(d['items'], d['mu_item']):
    assert abs(marg.get(tuple(X), 0.0) - m) < 1e-7, ('marginal', X)
s = pi.sum()
tol = 1e-8
worst = np.inf
# fixed-fixed
for u in range(6):
    for v in range(u + 1, 6):
        lhs = (nu * (FIX[:, u] != FIX[:, v])).sum()
        rhs = (pi * (canfix[:, u] != canfix[:, v])).sum()
        worst = min(worst, lhs - rhs)
# aux-fixed
for v in range(6):
    lhs = nu @ (COL != FIX[:, v][:, None])
    rhs = (DIG != canfix[:, v][None, :]) @ pi
    worst = min(worst, (lhs - rhs).min())
print('fixed pairs: worst slack', worst)
# aux-aux: lhs(f,g) = sum nu - sum_c sum_r nu_r [COL_rf = c][COL_rg = c];  rhs = s - sum_c sum_x pi_x [f_x=c][g_x=c]
Ec = [(COL == c).astype(np.float64) for c in range(K)]        # R x nF
Pc = [(DIG == c).astype(np.float64) for c in range(K)]        # nF x S
Ecn = [E * nu[:, None] for E in Ec]
B = 1500
for a in range(0, nF, B):
    sl = slice(a, min(a + B, nF))
    same_r = sum(Ecn[c][:, sl].T @ Ec[c] for c in range(K))          # B x nF
    same_p = sum((Pc[c][sl] * pi[None, :]) @ Pc[c].T for c in range(K))  # B x nF
    slack = (nu.sum() - same_r) - (s - same_p)
    worst = min(worst, slack.min())
print(f'all pairs: worst slack {worst:.3e}  (tol {tol})')
assert worst > -tol
print(f'CERTIFICATE OK: for every type-{t} gadget, E_mu V(x,y0) >= {s:.10f} * C')
