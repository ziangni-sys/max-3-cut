import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.jointlp import JointSymLP
from gadgets.symlp import SymDynLP
k = int(sys.argv[1])
d = pickle.load(open(f'results/reopt_nrt_k{k}.pkl', 'rb'))
cols = dict(d['groups'])
alphas = []
for i in range(k):
    lp = SymDynLP(k, i, cols[i], seed=0); g = lp.optimize(); alphas.append(1/g)
print('per-gadget alphas', [round(a, 4) for a in alphas], ' 1-(k-1)/sum =', 1 - (k-1)/sum(alphas))
for refined in [False, True]:
    t0 = time.time()
    J = JointSymLP(k, [cols[i] for i in range(k)], refined=refined, seed=0)
    tau = J.optimize()
    print(f'joint refined={refined}: tau={tau:.7f} rows={J.nrow} {time.time()-t0:.0f}s', flush=True)
