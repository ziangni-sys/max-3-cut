"""Sanity check: full TSSW column set for k=2 in the NRT framework (expect 16/17 = 0.941176)."""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from itertools import product
import numpy as np
from gadgets.search import *

k = 2
cols = list(product(range(k), repeat=k * k))
groups = make_single_group(k, cols)
print('vertices', groups[0].n, 'edges', groups[0].E)
lp = GadgetLP(groups, k, seed=1)
tau = lp.run(restarts=16)
print('LP tau =', tau)
w = lp.weights(0)
for scale in [12, 24, 1000]:
    wi = integerize(w, scale)
    mats = full_gadget_matrices(lp, [wi])
    ps, r = exact_params(mats, k, verbose=False)
    print('scale', scale, ps, 'exact ratio', r, float(r))
