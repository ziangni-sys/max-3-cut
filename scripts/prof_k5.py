import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.symlp import SymDynLP, generate_sym
st = pickle.load(open('results/s2_k5_i4_seed_cap28.ckpt', 'rb'))
cols = st['best']['cols']
lp = SymDynLP(5, 4, cols, seed=0)
print('n_aux', lp.n - 8, 'orbits', lp.norb, 'items', len(lp.items), flush=True)
lp.seed_random()
for it in range(40):
    t0 = time.time(); g = lp.solve(); ts = time.time() - t0
    t0 = time.time(); added, worst = lp.separate(tol=max(1e-7, 1e-3 * abs(g))); tp = time.time() - t0
    t0 = time.time(); pr = lp.prune_rows(); tq = time.time() - t0
    print(f'it {it}: g={g:.6f} solve={ts:.1f}s sep={tp:.1f}s prune={pr}({tq:.1f}s) added={added} worst={worst:.2g} rows={len(lp.row_col)} simplex_it={lp.h.getInfo().simplex_iteration_count}', flush=True)
    if added == 0: break
