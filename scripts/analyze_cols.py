"""Structure of the auxiliary columns of a gadget family: which linear forms they factor through."""
import sys
import os, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from collections import Counter
from gadgets.core import sat_assignments, edges_to_matrix
d = json.load(open(sys.argv[1])); k = d['k']
for g in d['gadgets']:
    i = g['i']; sat = sat_assignments(k, i)
    W = edges_to_matrix(g['edges'], n=g['n_vertices'])
    cols = g['columns']
    kinds = Counter()
    rows = []
    for u, f in enumerate(cols):
        f = np.array(f)
        # smallest set of linear forms L (up to scaling) with f constant on fibres of L
        facts = []
        for a in product(range(k), repeat=2):   # L = a0*x1 + a1*x2 (x3 determined)
            if a == (0, 0):
                continue
            L = np.array([(a[0] * x[0] + a[1] * x[1]) % k for x in sat])
            if all(len(set(f[L == v])) == 1 for v in set(L)):
                facts.append(a)
        nvals = len(set(f.tolist()))
        deg = int((W[3 + k + u] > 0).sum()) if 3 + k + u < W.shape[0] else 0
        key = 'const' if nvals == 1 else ('fn of one linear form' if facts else 'other')
        kinds[key] += 1
        rows.append((key, nvals, facts[:2], deg, int(W[3 + k + u].sum()) if 3 + k + u < W.shape[0] else 0))
    print(f'i={i}: {dict(kinds)}')
    for r in sorted(rows, key=lambda r: -r[4])[:8]:
        print('    ', r)
