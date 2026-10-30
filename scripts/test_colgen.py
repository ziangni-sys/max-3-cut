import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.colgen import *

k, i = 4, 0
d = pickle.load(open('results/reopt_nrt_k4.pkl', 'rb'))
cols0 = dict(d['groups'])[i]
lp = DynLP(k, i, cols0, seed=0)
t0 = time.time()
g = lp.optimize()
print('initial g', g, 'alpha', 1/g, f'{time.time()-t0:.0f}s rows', len(lp.row_col))
rc = lp.rc_existing()
W = lp.W()
pos = [v for (u, v_), v in [((e), rc[e]) for e in rc] if True]
vals = np.array(list(rc.values()))
wpos = np.array([W[u, v] > 1e-9 for (u, v) in rc])
print('RC max over all existing edges', vals.max(), ' RC on positive-weight edges: min', vals[wpos].min(), 'max', vals[wpos].max())
print('dual sums: pi', lp.duals()[0].sum(), 'y', lp.duals()[1].sum(), 'g', g)
for rnd in range(8):
    t1 = time.time()
    new = generate(lp, n_samples=2000, top=4)
    for rcv, f, fill, N, dd in new:
        lp.add_vertex(f, fill=fill)
    g = lp.optimize()
    print(f'round {rnd}: added {len(new)} (rc {[round(t[0],5) for t in new]}) -> g={g:.7f} alpha={1/g:.4f} n={lp.n} used={len(lp.used_vertices())} rows={len(lp.row_col)} {time.time()-t1:.0f}s', flush=True)
