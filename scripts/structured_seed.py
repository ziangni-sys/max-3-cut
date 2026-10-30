"""Seed columns: an auxiliary vertex restricted (via the globals) to a colour set S and
best-responding to all three primaries:  f(x) = argmin_{c in S} #{j : x_j = c}, ties broken
by a fixed order of S.  These are symmetric functions of x (S_3-invariant).
Usage: structured_seed.py k i [sizes=2,3] [max_orders]"""
import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import combinations, permutations
from gadgets.symlp import SymDynLP
from gadgets.core import sat_assignments

def seed_columns(k, i, sizes=(2, 3), max_orders=2):
    sat = sat_assignments(k, i)
    cols = []
    for s in sizes:
        for S in combinations(range(k), s):
            for order in list(permutations(S))[:max_orders]:
                f = []
                for x in sat:
                    cnt = {c: sum(1 for v in x if v == c) for c in order}
                    best = min(cnt.values())
                    f.append(next(c for c in order if cnt[c] == best))
                cols.append(tuple(f))
    return list(dict.fromkeys(cols))

if __name__ == '__main__':
    k, i = int(sys.argv[1]), int(sys.argv[2])
    sizes = tuple(int(t) for t in sys.argv[3].split(',')) if len(sys.argv) > 3 else (2, 3)
    mo = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    cols = seed_columns(k, i, sizes, mo)
    t0 = time.time()
    lp = SymDynLP(k, i, cols, seed=0)
    print(f'k={k} i={i}: {len(cols)} seed columns, n_aux={lp.n-3-k}, orbits={lp.norb}', flush=True)
    g = lp.optimize(rel_tol=1e-3)
    used = lp.used_vertices()
    print(f'  g={g:.6f} alpha={1/g if g > 0 else float("inf"):.3f} used={len(used)} rows={len(lp.row_col)} {time.time()-t0:.0f}s', flush=True)
    pickle.dump({'cols': [lp.vcols[u] for u in used] if g > 0 else cols, 'g': g},
                open(f'{_ROOT}/results/struct_seed_k{k}_i{i}_s{"".join(map(str, sizes))}_o{mo}.pkl', 'wb'))
