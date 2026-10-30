import sys
import os, json, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from gadgets.core import edges_to_matrix, allowed_global_assignments
from gadgets.sdpbound import sdp_upper_bound
from gadgets.localsearch import batched_local_search
d = json.load(open('results/fam_k5_c32.json')); k = 5
g = d['gadgets'][0]; i = g['i']
W = edges_to_matrix(g['edges'], n=g['n_vertices'])
Y = allowed_global_assignments(k)
reps = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3)))
keys = [(x, y) for x in reps for y in Y]
fixed = np.array([list(x) + list(y) for x, y in keys])
lv, _ = batched_local_search(W.astype(float), k, fixed, 3 + k, restarts=16, rounds=4)
Ts = max(v for (x, y), v in zip(keys, lv) if sum(x) % k == i)
Tu = max(v for (x, y), v in zip(keys, lv) if sum(x) % k != i)
print('global thresholds: sat', Ts, 'unsat', Tu)
rng = np.random.default_rng(3)
cert = 0; tot = 0; gaps = []
for j in rng.choice(len(keys), 40, replace=False):
    x, y = keys[j]
    T = Ts if sum(x) % k == i else Tu
    t0 = time.time()
    ub, pv, st = sdp_upper_bound(W, k, list(x) + list(y), 3 + k)
    ok = ub <= T
    cert += ok; tot += 1
    gaps.append((ub - pv) / max(pv, 1))
    print(x, y, f'LS={lv[j]:.0f} T={T:.0f} SDPprimal={pv:.0f} rigorousUB={ub:.0f} {"CERT" if ok else "--"} {st} {time.time()-t0:.1f}s', flush=True)
print(f'certified {cert}/{tot}; max relative (rigorous - primal) = {max(gaps):.2e}')
