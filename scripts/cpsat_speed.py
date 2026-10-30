import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.cpsat import cpsat_max
st = pickle.load(open(sys.argv[1], 'rb'))
W = st['best']['W']; k = st['k']
Wi = np.rint(W / W.max() * 1e5).astype(np.int64)
print('n_aux', W.shape[0] - 3 - k, 'nnz edges', int((np.triu(Wi, 1) > 0).sum()))
rng = np.random.default_rng(0)
for trial in range(4):
    x = list(rng.integers(0, k, 3)); y = [0, 1, 2, 3] if trial % 2 == 0 else [0, 1, 1, 2]
    t0 = time.time(); v, c = cpsat_max(Wi, k, x + y, 3 + k, workers=1, time_limit=600)
    print(x, y, v, f'{time.time()-t0:.2f}s', flush=True)
