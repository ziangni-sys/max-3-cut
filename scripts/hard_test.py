import sys
import os, json, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from gadgets.core import edges_to_matrix, allowed_global_assignments
from gadgets.cpsat import cpsat_decide, cpsat_decide_strong
from gadgets.localsearch import batched_local_search
d = json.load(open('results/fam_k4_v1.json')); k = 4
g = d['gadgets'][0]; i = 0
W = edges_to_matrix(g['edges'], n=g['n_vertices'])
y0 = (0, 1, 2, 3)
X = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3)))
uns = [x for x in X if sum(x) % k != i]
fixed = np.array([list(x) + list(y0) for x in uns])
lv, lc = batched_local_search(W.astype(float), k, fixed, 7, restarts=24, rounds=5)
T = int(np.rint(lv.max()))
j = uns.index((0, 0, 2))
print('threshold', T, 'LS at (0,0,2):', lv[j])
for name, fn in [('strong', cpsat_decide_strong), ('basic', cpsat_decide)]:
    t0 = time.time()
    try:
        r = fn(W, k, list(fixed[j]), 7, T, time_limit=float(sys.argv[1]), hint=lc[j])
        print(name, 'result', None if r is None else r[0], f'{time.time()-t0:.0f}s', flush=True)
    except RuntimeError as e:
        print(name, e, f'{time.time()-t0:.0f}s', flush=True)
