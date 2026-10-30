"""Greedy orbit pruning of a converged single-predicate gadget, re-optimizing to convergence
after each removal.  Saves every stage.  Usage: prune_state.py state.pkl out.pkl alpha_max min_aux"""
import sys
import os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.symlp import SymDynLP

st = pickle.load(open(sys.argv[1], 'rb'))
out, amax, min_aux = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
k, i = st['k'], st['i']
nf = 3 + k
cols = [tuple(c) for c in st['used_cols']]

def converge(lp):
    while True:
        g = lp.optimize(max_iter=2000, rel_tol=1e-4, max_rows=8000)
        a, w = lp.separate(restarts=16, rounds=6, tol=max(1e-7, 1e-4 * abs(g)))
        if a == 0:
            return lp.solve()

def build(cols, rows):
    lp = SymDynLP(k, i, cols, seed=1)
    pos = {c: u for u, c in enumerate(lp.vcols)}
    old = [tuple(c) for c in st['vcols']]
    rc, rs = [], []
    for col, sat in rows:
        m = {old[u]: col[u] for u in range(len(old))} if len(col) == len(old) else None
        if m is None:
            continue
        rc.append([m.get(f, 0) for f in lp.vcols])
        rs.append(sat)
    if rc:
        lp.add_sound_rows(rc, rs)
    else:
        lp.seed_random()
    return lp

stages = []
t0 = time.time()
lp = build(cols, list(zip(st['rows'], st['row_sat'])))
g = converge(lp)
print(f'start: alpha={1/g:.3f} n_aux={len(lp.used_vertices())} {time.time()-t0:.0f}s', flush=True)
while True:
    used = lp.used_vertices()
    W = lp.W()
    stages.append(dict(g=float(g), alpha=1 / g, n_aux=len(used), cols=[lp.vcols[u] for u in used],
                       W=W[np.ix_(list(range(nf)) + used, list(range(nf)) + used)].copy(), k=k, i=i))
    pickle.dump(stages, open(out, 'wb'))
    if len(used) <= min_aux:
        break
    orbs = {}
    for u in used:
        orbs.setdefault(min(lp.orbit_cols(lp.vcols[u])), []).append(u)
    worst = min(orbs.values(), key=lambda us: W[us].sum())
    lp.kill_orbit(worst[0])
    g = converge(lp)
    print(f'removed orbit of size {len(worst)}: alpha={1/g:.3f} n_aux={len(lp.used_vertices())} {time.time()-t0:.0f}s', flush=True)
    if 1 / g > amax:
        break
print('done', flush=True)
