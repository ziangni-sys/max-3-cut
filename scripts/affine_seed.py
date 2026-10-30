"""LP value of the S_3-symmetric gadget LP when the auxiliary columns are all affine
functions a1*x1 + a2*x2 + c (mod k) on the satisfying set (x3 determined)."""
import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from gadgets.symlp import SymDynLP
from gadgets.core import sat_assignments
k, i = int(sys.argv[1]), int(sys.argv[2])
sat = sat_assignments(k, i)
cols = []
for a1, a2, c in product(range(k), repeat=3):
    cols.append(tuple((a1 * x[0] + a2 * x[1] + c) % k for x in sat))
t0 = time.time()
lp = SymDynLP(k, i, cols, seed=0)
print('n_aux', lp.n - 3 - k, 'orbits', lp.norb, flush=True)
g = lp.optimize(rel_tol=1e-3, verbose=False)
used = lp.used_vertices()
print(f'k={k} i={i}: g={g:.6f} alpha={1/g if g > 0 else float("inf"):.3f} used={len(used)} rows={len(lp.row_col)} {time.time()-t0:.0f}s', flush=True)
pickle.dump({'cols': [lp.vcols[u] for u in used], 'g': g}, open(f'results/affine_seed_k{k}_i{i}.pkl', 'wb'))
