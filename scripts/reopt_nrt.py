"""Re-optimize the weights of NRT's gadgets by LP, keeping their witness columns,
but with an independent gadget for every predicate i."""
import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.core import *
from gadgets.search import *
from gadgets.nrt_gadgets import *

k = int(sys.argv[1])
if k == 3:
    G0 = edges_to_matrix(NRT_K3_G0, n=14); G1 = edges_to_matrix(NRT_K3_G1, n=14)
    mats = [G0, G1, relabel_globals(G1, 3, [0, 2, 1])]
else:
    H = [edges_to_matrix(NRT_K4_G0, n=19)]
    for i in range(1, 4):
        H.append(relabel_globals(H[-1], 4, [3, 0, 1, 2]))
    mats = H
groups = []
for i in range(k):
    cols = witness_columns(mats[i], k, i)
    groups.append(Group(k, i, cols))
    print(f'i={i}: {len(groups[-1].columns)} distinct aux columns', flush=True)
lp = GadgetLP(groups, k, seed=0)
t0 = time.time()
tau = lp.run(restarts=8, max_iter=300, log_every=5)
print('heuristic LP tau', tau, f'{time.time()-t0:.0f}s', flush=True)
tau = run_exact(lp)
print('exact-separation LP tau', tau, flush=True)
pickle.dump({'groups': [(g.base_i, g.columns) for g in groups], 'w': [lp.weights(gi) for gi in range(k)]},
            open(f'{_ROOT}/results/reopt_nrt_k{k}.pkl', 'wb'))
for scale in [1000, 20000]:
    ps, r = exact_params(full_gadget_matrices(lp, integerize_all(lp, scale)), k, verbose=False)
    print('scale', scale, ps, 'EXACT ratio', r, float(r), flush=True)
