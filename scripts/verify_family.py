"""Exact verification of a gadget family (JSON, NRT edge-list format).
Computes V_i[x][y] = max_z I_i(x,y,z) for all x in Z_k^3, y in Y by MITM (small) or CP-SAT
(proved optimal), then  C_i, S1_i, S2_i, the NRT ratio (Thm A.4) and the shared-y refined ratio.
Usage: verify_family.py family.json [workers]"""
import sys
import os, json, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from fractions import Fraction
from itertools import product
import numpy as np
from gadgets.core import edges_to_matrix, allowed_global_assignments, FixedMaxCut
from gadgets.cpsat import cpsat_max

path = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
d = json.load(open(path))
k = d['k']
Y = allowed_global_assignments(k)
y0 = tuple(range(k))
X = list(product(range(k), repeat=3))
tables = []
for g in d['gadgets']:
    i = g['i']
    W = edges_to_matrix(g['edges'], n=g['n_vertices'])
    naux = W.shape[0] - 3 - k
    t0 = time.time()
    if k ** ((naux + 1) // 2) <= 5000:
        fm = FixedMaxCut(W, k)
        mx = lambda x, y: fm.maxval(x, y)
        meth = 'mitm'
    else:
        mx = lambda x, y: cpsat_max(W, k, list(x) + list(y), 3 + k, time_limit=3600, workers=workers)[0]
        meth = 'cpsat'
    V = {(x, y): mx(x, y) for x in X for y in Y}
    C = min(V[(x, y0)] for x in X if sum(x) % k == i)
    S1y = {y: max(V[(x, y)] for x in X if sum(x) % k == i) for y in Y}
    S2y = {y: max(V[(x, y)] for x in X if sum(x) % k != i) for y in Y}
    t = int(np.triu(W, 1).sum())
    tables.append(dict(i=i, C=C, S1=max(S1y.values()), S2=max(S2y.values()), t=t, S1y=S1y, S2y=S2y))
    print(f'i={i}: C={C} S1={max(S1y.values())} S2={max(S2y.values())} t={t} n_aux={naux} [{meth}, {time.time()-t0:.0f}s]', flush=True)
sumC = sum(p['C'] for p in tables)
nrt = Fraction(sum(p['S2'] for p in tables) + max(p['S1'] - p['S2'] for p in tables), sumC)
ref_num = max(sum(p['S2y'][y] for p in tables) + max(0, max(p['S1y'][y] - p['S2y'][y] for p in tables)) for y in Y)
ref = Fraction(ref_num, sumC)
print(f'NRT Thm A.4 ratio      = {nrt} = {float(nrt):.7f}')
print(f'shared-y refined ratio = {ref} = {float(ref):.7f}')
json.dump({'nrt_ratio': [nrt.numerator, nrt.denominator], 'refined_ratio': [ref.numerator, ref.denominator],
           'gadgets': [{'i': p['i'], 'C': p['C'], 'S1': p['S1'], 'S2': p['S2'], 't': p['t'],
                        'S1y': {''.join(map(str, y)): v for y, v in p['S1y'].items()},
                        'S2y': {''.join(map(str, y)): v for y, v in p['S2y'].items()}} for p in tables]},
          open(path.replace('.json', '.verified.json'), 'w'), indent=1)
