"""Column local search for one predicate gadget. Usage: colsearch_run.py k i steps seed [max_cols]"""
import sys
import os, time, pickle, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.colsearch import *

k, i, steps, seed = map(int, sys.argv[1:5])
max_cols = int(sys.argv[5]) if len(sys.argv) > 5 else 20
init = sys.argv[6] if len(sys.argv) > 6 else f'{_ROOT}/results/reopt_nrt_k{k}.pkl'
d = pickle.load(open(init, 'rb'))
cols0 = dict(d['groups'])[i] if 'groups' in d else d['cols']
cs = ColumnSearch(k, i, cols0, seed=seed, max_cols=max_cols)
out = f'{_ROOT}/results/cs_k{k}_i{i}_s{seed}.pkl'
t0 = time.time()
for s in range(steps):
    g, nc, dropped, nrows = cs.step(nprop=6)
    print(f'step {s}: g={g:.7f} alpha={1/g:.4f} cols={nc} dropped={dropped} rows={nrows} best_alpha={1/cs.best[0]:.4f} t={time.time()-t0:.0f}s', flush=True)
    if s % 5 == 4 or s == steps - 1:
        pickle.dump({'cols': cs.best[1], 'w': cs.best[2], 'g': cs.best[0], 'k': k, 'i': i,
                     'cur_cols': cs.cols}, open(out, 'wb'))
