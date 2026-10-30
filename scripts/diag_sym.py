import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.symlp import *
d = pickle.load(open('results/reopt_nrt_k4.pkl', 'rb'))
cols = dict(d['groups'])[0]
lp = SymDynLP(4, 0, cols, seed=0)
g = lp.optimize()
print('alpha', 1/g, 'rows', len(lp.row_col), flush=True)
for rnd in range(6):
    t1 = time.time()
    new = generate_sym(lp)
    for rc, f, fills, N, dd in new:
        lp.add_orbit(f, fills=fills)
    t2 = time.time()
    for it in range(300):
        ts = time.time(); g = lp.solve(); tsol = time.time() - ts
        ts = time.time(); added, worst = lp.separate(); tsep = time.time() - ts
        print(f'   it {it}: g={g:.7f} added={added} rows={len(lp.row_col)} orbits={lp.norb} solve={tsol:.2f}s sep={tsep:.2f}s iters={lp.h.getInfo().simplex_iteration_count}', flush=True)
        if added == 0: break
    print(f'round {rnd}: alpha={1/g:.4f} n={lp.n} gen={t2-t1:.1f}s total={time.time()-t1:.1f}s', flush=True)
