import sys
import os, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from gadgets.core import edges_to_matrix, allowed_global_assignments
from gadgets.localsearch import batched_local_search
d = json.load(open('results/fam_k4_v1.json')); k = 4
Y = allowed_global_assignments(k)
g = d['gadgets'][0]; i = g['i']
W = edges_to_matrix(g['edges'], n=g['n_vertices']).astype(float)
X = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3)))
keys = [(x, y) for x in X for y in Y]
fixed = np.array([list(x) + list(y) for x, y in keys])
for R, rounds in [(8, 3), (24, 5)]:
    v, c = batched_local_search(W, k, fixed, 3 + k, restarts=R, rounds=rounds)
    sat = np.array([sum(x) % k == i for x, _ in keys])
    print(f'R={R}: max sat {v[sat].max():.0f}  max unsat {v[~sat].max():.0f}')
    top = np.argsort(-np.where(sat, -1, v))[:5]
    for j in top: print('   unsat', keys[j], v[j])
# also full x (no symmetry) at the worst y
