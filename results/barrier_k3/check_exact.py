"""EXACT check of a barrier certificate (mudual_t*.pkl), cf. Lemma 7.2 of the paper.

The LP solution (pi, nu) is rationalised (all entries have small denominators); with D the common
denominator, PI = D*pi and NU = D*nu are integers, and every constraint
    sum_r NU_r [row r cuts e]  >=  sum_x PI_x [canonical colouring at x cuts e]
for every pair e of the universe (3 primaries, 3 globals, all 3^9 columns on sat_t) is checked in
integer-valued float64 arithmetic (all partial sums < 2^53, hence exact).  Also checks exactly:
sum NU = D, the x-marginals of nu, rows at y0 with unsatisfying x, and sum(pi) = target.
Usage: check_exact.py cert.pkl target   (e.g. 8/9)
"""
import sys, pickle
from fractions import Fraction as F
from math import lcm
from itertools import product
import numpy as np

if sys.argv[1].endswith('.json'):
    import json
    _j = json.load(open(sys.argv[1]))
    _D = _j['D']
    d = {'t': _j['type'], 'pi': [F(v, _D) for v in _j['PI']], 'nu': [F(r['NU'], _D) for r in _j['rows']],
         'rows': [('L', tuple(r['T']), tuple(r['h']), tuple(r['x']), (0, 1, 2)) for r in _j['rows']]}
    _xs = sorted({tuple(r['x']) for r in _j['rows']})
    from itertools import permutations as _perm
    _orb = sorted({tuple(sorted(x)) for x in _xs})
    _m = {}
    for r in _j['rows']:
        _m[tuple(sorted(r['x']))] = _m.get(tuple(sorted(r['x'])), F(0)) + F(r['NU'], _D)
    d['items'] = [list(o) for o in _orb]; d['mu_item'] = [_m[o] for o in _orb]
else:
    d = pickle.load(open(sys.argv[1], 'rb'))
target = F(sys.argv[2])
t, rows = d['t'], d['rows']
K = 3
sat = [x for x in product(range(K), repeat=3) if sum(x) % K == t]
S = len(sat)
DIG = np.array(list(product(range(K), repeat=S)), dtype=np.int64)
nF = len(DIG)
y0 = (0, 1, 2)
canfix = np.array([list(x) + list(y0) for x in sat])

def rat(v):
    if isinstance(v, F):
        return v
    q = F(float(v)).limit_denominator(10 ** 6)
    assert abs(float(q) - float(v)) < 1e-9, (v, q)
    return q

nuq = [rat(v) for v in d['nu']]
piq = [rat(v) for v in d['pi']]
keep = [i for i, q in enumerate(nuq) if q > 0]
rows = [rows[i] for i in keep]
nuq = [nuq[i] for i in keep]
D = lcm(*[q.denominator for q in nuq + piq if q != 0])
NU = np.array([int(q * D) for q in nuq], dtype=np.float64)
PI = np.array([int(q * D) for q in piq], dtype=np.float64)
assert all(float(int(q * D)) == q * D for q in nuq + piq)
print(f'rows {len(rows)}  common denominator D = {D}  sum nu = {F(int(NU.sum()), D)}  sum pi = {F(int(PI.sum()), D)}')
assert int(NU.sum()) == D, 'nu must sum to exactly 1'
assert F(int(PI.sum()), D) == target, 'sum(pi) != target'
assert all(tuple(r[4]) == y0 for r in rows) and all(sum(r[3]) % K != t for r in rows)
assert D * len(rows) < 2 ** 50

def colour(row):
    _, T, h, x, y = row
    assert row[0] == 'L' and len(T) <= 2
    pw = K ** np.arange(len(T))
    return np.asarray(h)[(DIG[:, list(T)] * pw[None, :]).sum(1)]

COL = np.array([colour(r) for r in rows])
FIX = np.array([list(r[3]) + list(r[4]) for r in rows])
assert all(q >= 0 for q in piq), 'pi must be non-negative'
# per-tuple x-marginal of nu (not only per S_3-orbit): must equal the prescribed orbit mass spread
# uniformly over the tuples of the orbit (classes: all-equal, all-distinct, two-equal by sum)
marg = {}
for r, q in zip(rows, nuq):
    marg[tuple(r[3])] = marg.get(tuple(r[3]), 0) + q
mu = {tuple(X): rat(m) for X, m in zip(d['items'], d['mu_item'])}
from itertools import permutations
for X, m in mu.items():
    orbit = sorted(set(permutations(X)))
    for x in orbit:
        assert marg.get(x, 0) == m / len(orbit), ('per-tuple marginal', x, marg.get(x, 0), m / len(orbit))
assert sum(marg.values()) == 1 and set(marg) <= {x for X in mu for x in permutations(X)}
def cls(x):
    return 'A' if len(set(x)) == 1 else 'D' if len(set(x)) == 3 else 'T%d' % (sum(x) % 3)
cm = {}
for x, q in marg.items():
    cm[cls(x)] = cm.get(cls(x), 0) + q
print('x-marginal of nu per class (exact, uniform within each class):', {c: str(q) for c, q in sorted(cm.items())})
worst = None
for u in range(6):
    for v in range(u + 1, 6):
        s_ = (NU * (FIX[:, u] != FIX[:, v])).sum() - (PI * (canfix[:, u] != canfix[:, v])).sum()
        worst = s_ if worst is None else min(worst, s_)
for v in range(6):
    s_ = (NU @ (COL != FIX[:, v][:, None])) - ((DIG != canfix[:, v][None, :]) @ PI)
    worst = min(worst, s_.min())
print('fixed pairs: min slack (units of 1/D):', worst)
Ec = [(COL == c).astype(np.float64) for c in range(K)]
Pc = [(DIG == c).astype(np.float64) for c in range(K)]
Ecn = [E * NU[:, None] for E in Ec]
tot_nu, tot_pi = NU.sum(), PI.sum()
B = 1500
for a in range(0, nF, B):
    sl = slice(a, min(a + B, nF))
    same_r = sum(Ecn[c][:, sl].T @ Ec[c] for c in range(K))
    same_p = sum((Pc[c][sl] * PI[None, :]) @ Pc[c].T for c in range(K))
    worst = min(worst, ((tot_nu - same_r) - (tot_pi - same_p)).min())
print('all pairs: min slack (units of 1/D):', worst)
assert worst >= 0 and float(worst).is_integer()
print(f'EXACT CERTIFICATE OK: every type-{t} gadget has E_nu V_t(x, y0) >= {target} * C_t')
