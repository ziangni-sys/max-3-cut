import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from itertools import product
import numpy as np
from gadgets.symlp import *
from gadgets.colgen import DynLP

# k=2 full columns: expect alpha 8 (i=0) and 9 (i=1)
for i in range(2):
    cols = list(product(range(2), repeat=4))
    lp = SymDynLP(2, i, cols, seed=0)
    g = lp.optimize()
    print(f'k=2 i={i}: sym alpha={1/g:.6f} orbits={lp.norb} n={lp.n}')
# k=3 reopt columns, sym-closed vs plain
d = pickle.load(open('results/reopt_nrt_k3.pkl', 'rb'))
for i, cols in d['groups']:
    t0 = time.time()
    lp = SymDynLP(3, i, cols, seed=0); g = lp.optimize()
    lp2 = DynLP(3, i, cols, seed=0); g2 = lp2.optimize()
    print(f'k=3 i={i}: plain alpha={1/g2:.5f} (n={lp2.n})  sym-closed alpha={1/g:.5f} (n={lp.n}, orbits={lp.norb}) {time.time()-t0:.0f}s')
