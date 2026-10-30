import sys
import os, json, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from gadgets.core import edges_to_matrix, allowed_global_assignments
from gadgets.cpsat import cpsat_max
from gadgets.localsearch import batched_local_search
d = json.load(open(sys.argv[1])); k = d['k']
Y = allowed_global_assignments(k)
rng = np.random.default_rng(1)
for g in d['gadgets']:
    W = edges_to_matrix(g['edges'], n=g['n_vertices'])
    keys = [(tuple(rng.integers(0, k, 3)), Y[rng.integers(len(Y))]) for _ in range(8)]
    fixed = np.array([list(x) + list(y) for x, y in keys])
    _, hints = batched_local_search(W.astype(float), k, fixed, 3 + k, restarts=16, rounds=4)
    ts = []
    for f, h in zip(fixed, hints):
        t0 = time.time(); cpsat_max(W, k, list(f), 3 + k, time_limit=600, hint=h); ts.append(time.time() - t0)
    print(f"i={g['i']} n_aux={W.shape[0]-3-k} times: mean={np.mean(ts):.2f}s max={np.max(ts):.2f}s", flush=True)
