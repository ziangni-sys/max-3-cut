"""Assemble the best per-predicate gadgets into a family, round jointly, verify exactly.
Usage: assemble.py k scale out.json ckpt_0 ckpt_1 ... ckpt_{k-1}"""
import sys
import os, time, pickle, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from fractions import Fraction
from gadgets.core import gadget_params, gadget_params_cpsat, hardness_ratio

k, scale, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
ck = sys.argv[4:4 + k]
Ws, gs = [], []
for i, c in enumerate(ck):
    st = pickle.load(open(c, 'rb'))
    b = st['best']
    W = b['W'] / b['g']          # gap normalized to 1 (completeness ~ alpha_i)
    Ws.append(W)
    gs.append(b['g'])
    print(f'i={i}: LP alpha={1/b["g"]:.5f}  n_aux={W.shape[0]-3-k}', flush=True)
print('LP-predicted ratio', 1 - (k - 1) / sum(1 / g for g in gs), flush=True)
m = max(W.max() for W in Ws)
mats = [np.rint(W / m * scale).astype(np.int64) for W in Ws]
ps = []
for i, W in enumerate(mats):
    keep = [u for u in range(W.shape[0]) if u < 3 + k or W[u].sum() > 0]
    W = W[np.ix_(keep, keep)]
    mats[i] = W
    t0 = time.time()
    naux = W.shape[0] - 3 - k
    p = gadget_params(W, k, i) if k ** ((naux + 1) // 2) <= 5000 else gadget_params_cpsat(W, k, i)
    ps.append(p)
    print(f'  exact i={i}: {p} n_aux={naux} ({time.time()-t0:.0f}s)', flush=True)
r = hardness_ratio(ps)
print('EXACT ratio', r, float(r), flush=True)
def edges(W):
    n = W.shape[0]
    return [(u + 1, v + 1, int(W[u, v])) for u in range(n) for v in range(u + 1, n) if W[u, v]]
json.dump({'k': k, 'ratio': [r.numerator, r.denominator], 'ratio_float': float(r),
           'gadgets': [{'i': i, 'n_vertices': int(W.shape[0]), 'edges': edges(W), 'params': p}
                       for i, (W, p) in enumerate(zip(mats, ps))]}, open(out, 'w'), indent=1)
print('saved', out)
