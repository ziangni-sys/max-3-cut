"""Independent check of the 'negation merge' reduction 3LIN(3) -> MAX-3-CUT (track T, Thm E).
E-gadget (type 0, x1+x2+x3=0 <=> all equal or all distinct): NRT's I_0 (no globals), weights x2.
N-gadget (types 1,2, sum != 0 <=> exactly two equal): triangle on primaries (w=1) + one aux
vertex joined to each primary with weight 3.
Claim: for every x, J(x) = Base + 2*(#sat(x) + #sat(-x)) exactly, hence ratio 27/29 (balanced)."""
import sys
import os, itertools, random
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.core import edges_to_matrix
from gadgets.nrt_gadgets import NRT_K3_G0

k = 3
G0 = edges_to_matrix(NRT_K3_G0, n=14)          # vertices 1..3 primary, 4..6 global (unused), 7..12 aux

def gadget_max(W, x):
    n = W.shape[0]; aux = [u for u in range(3, n) if W[u].sum() > 0 and u >= 6] if False else list(range(6, n))
    best = -1
    for z in itertools.product(range(k), repeat=len(aux)):
        col = np.zeros(n, dtype=int); col[:3] = x; col[3:6] = [0, 1, 2]
        col[aux] = z
        best = max(best, int((W * (col[:, None] != col[None, :])).sum() // 2))
    return best

assert all(G0[u, v] == 0 for u in range(3, 6) for v in range(14)), 'G0 uses globals?'
E = {x: 2 * gadget_max(G0, x) for x in itertools.product(range(3), repeat=3)}
def N(x):
    tri = sum(1 for a, b in [(0, 1), (1, 2), (0, 2)] if x[a] != x[b])
    return tri + 3 * max(sum(1 for v in x if v != w) for w in range(3))
Nv = {x: N(x) for x in itertools.product(range(3), repeat=3)}
print('E values by pattern:', sorted(set((('sum0' if sum(x) % 3 == 0 else 'sum!=0'), v) for x, v in E.items())))
print('N values by pattern:', sorted(set((('sum0' if sum(x) % 3 == 0 else 'sum!=0'), v) for x, v in Nv.items())))
# per-constraint identity: type0: E(x)=32+4[s=0];  type1/2: N(x)=9+2[s!=0]
assert all(E[x] == 32 + 4 * (sum(x) % 3 == 0) for x in E)
assert all(Nv[x] == 9 + 2 * (sum(x) % 3 != 0) for x in Nv)
# random 3LIN(3) instances: check J(x) == Base + 2(sat(x)+sat(-x)) for all assignments
rng = random.Random(1)
for trial in range(200):
    nvar = rng.randint(3, 7); m = rng.randint(1, 12)
    cons = [(tuple(rng.sample(range(nvar), 3)), rng.randrange(3)) for _ in range(m)]
    for X in itertools.product(range(3), repeat=nvar):
        J = sum(E[tuple(X[v] for v in vs)] if t == 0 else Nv[tuple(X[v] for v in vs)] for vs, t in cons)
        sat = sum((sum(X[v] for v in vs) - t) % 3 == 0 for vs, t in cons)
        satn = sum((sum(-X[v] for v in vs) - t) % 3 == 0 for vs, t in cons)
        base = sum(32 if t == 0 else 9 for vs, t in cons)
        assert J == base + 2 * (sat + satn), (cons, X)
print('identity J(x) = Base + 2(sat(x)+sat(-x)) verified on 200 random instances, all assignments')
from fractions import Fraction as F
bal = F(32 + 2 * 9 + 4, 36 + 2 * 11)          # (m/3 units): Base 32+18, + 4*(1/3)*3 ; completeness 36+22
print('balanced ratio =', bal, float(bal), ' 16/17 =', float(F(16, 17)))
worst = max(F(31, 3) + 23 * w, 0) / (11 + 25 * w) if False else None
ws = [F(j, 100) for j in range(101)]
print('unbalanced worst ratio over type-0 fraction w0 =', max((F(9) + 23 * w + F(4, 3)) / (11 + 25 * w) for w in ws), '(at w0=0: 31/33 =', float(F(31, 33)), ')')
