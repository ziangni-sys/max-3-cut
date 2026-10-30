"""Exact verification of the negation-merge reduction 3LIN(3) -> MAX-3-CUT (results/sym3_k3.json).

Self-contained (no solver): brute force over all primary and auxiliary colourings, integer arithmetic.
Checks (1) gadget profiles E(x) = 32 + 4[s=0], N(x) = 9 + 2[s!=0] with s = x1+x2+x3 mod 3, globals
irrelevant (no edges at vertices 4-6); (2) the per-equation identity value_t(x) = Base_t + 2([s=t]+[s=-t]);
(3) the resulting ratios: balanced 27/29, worst unbalanced 31/33.
Usage: verify_sym3.py [results/sym3_k3.json]
"""
import sys, json
from itertools import product
from fractions import Fraction as F

path = sys.argv[1] if len(sys.argv) > 1 else 'results/sym3_k3.json'
d = json.load(open(path))
G = {g['name']: g for g in d['gadgets']}


def require(c, msg):
    if not c:
        raise ValueError(msg)


def profile(g):
    n, edges = g['n_vertices'], g['edges']
    require(all(isinstance(w, int) and w > 0 and a != b for a, b, w in edges), 'bad edge')
    require(all(a not in (4, 5, 6) and b not in (4, 5, 6) for a, b, _ in edges), 'gadget must not use globals')
    aux = sorted({v for a, b, _ in edges for v in (a, b) if v >= 7})
    prof = {}
    for x in product(range(3), repeat=3):
        best = -1
        for z in product(range(3), repeat=len(aux)):
            col = {1: x[0], 2: x[1], 3: x[2]}
            col.update(zip(aux, z))
            best = max(best, sum(w for a, b, w in edges if col[a] != col[b]))
        prof[x] = best
    return prof, sum(w for *_, w in edges)


PE, tE = profile(G['E'])
PN, tN = profile(G['N'])
for x in PE:
    s = sum(x) % 3
    require(PE[x] == 32 + 4 * (s == 0), f'E profile at {x}')
    require(PN[x] == 9 + 2 * (s != 0), f'N profile at {x}')
    for t in range(3):                      # value_t(x) = Base_t + 2([s=t] + [s=-t])
        val, base = (PE[x], 32) if t == 0 else (PN[x], 9)
        require(val == base + 2 * ((s == t) + (s == (-t) % 3)), f'identity t={t} x={x}')
print(f'E: total weight {tE}, value 36 on sum=0, 32 otherwise;  N: total weight {tN}, value 11 on sum!=0, 9 otherwise')
print('identity value_t(x) = Base_t + 2([s=t]+[s=-t]) holds for all x, t')
# soundness: #sat(X) + #sat(-X) <= 2(1/3+eps)m  =>  J <= Base + 4m/3 ;  completeness: C_E = 36, C_N = 11
def ratio(w0):                              # w0 = fraction of type-0 equations
    snd = 32 * w0 + 9 * (1 - w0) + min(F(4, 3), 2 * (2 * w0 + (1 - w0)))
    return snd / (36 * w0 + 11 * (1 - w0))
bal = ratio(F(1, 3))
worst = max(ratio(F(j, 1000)) for j in range(1001))
require(bal == F(27, 29) and worst == F(31, 33), 'ratio')
print(f'balanced ratio {bal} = {float(bal):.6f}; worst over type-0 fraction {worst} = {float(worst):.6f} (16/17 = {16/17:.6f})')
print(f'gap: completeness {F(29, 30)} vs soundness {F(9, 10)} of the total edge weight (balanced)')
