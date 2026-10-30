"""k=2, full column set, independent gadgets per predicate (expect TSSW's 16/17)."""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from itertools import product
import numpy as np
from gadgets.search import *

k = 2
cols = list(product(range(k), repeat=k * k))
groups = [Group(k, 0, cols), Group(k, 1, cols)]
lp = GadgetLP(groups, k, seed=1, verbose=False)
tau = lp.run(restarts=16)
print('LP tau =', tau, ' 16/17 =', 16/17)
for scale in [24, 1000]:
    mats = full_gadget_matrices(lp, integerize_all(lp, scale))
    ps, r = exact_params(mats, k, verbose=False)
    print('scale', scale, ps, 'exact ratio', r, float(r))
