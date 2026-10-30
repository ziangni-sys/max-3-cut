import sys
import os, time, pickle, cProfile, pstats
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.symlp import *
d = pickle.load(open('results/reopt_nrt_k4.pkl', 'rb'))
cols = dict(d['groups'])[0]
t0 = time.time()
lp = SymDynLP(4, 0, cols, seed=0)
print('n', lp.n, 'orbits', lp.norb, f'{time.time()-t0:.1f}s')
pr = cProfile.Profile(); pr.enable()
g = lp.optimize(verbose=False)
print('alpha', 1/g, 'rows', len(lp.row_col), f'{time.time()-t0:.1f}s')
t1 = time.time()
new = generate_sym(lp)
print('gen', len(new), f'{time.time()-t1:.1f}s')
for rc, f, fills, N, dd in new:
    lp.add_orbit(f, fills=fills)
t2 = time.time(); g = lp.optimize(); print('alpha', 1/g, 'n', lp.n, 'orbits', lp.norb, 'rows', len(lp.row_col), f'{time.time()-t2:.1f}s')
pr.disable(); pstats.Stats(pr).sort_stats('cumulative').print_stats(12)
