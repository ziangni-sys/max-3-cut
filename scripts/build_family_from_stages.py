"""Integer family JSON (columns + S_3 automorphisms) from pruning stages (prune_state.py output).
Picks, for each predicate, the best stage with n_aux <= cap.  Each gadget is scaled to unit gap.
Usage: build_family_from_stages.py k cap scale out.json prune_0.pkl ... prune_{k-1}.pkl"""
import sys
import os, json, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.core import sat_assignments
from gadgets.symlp import PERMS, act_x, inv

k, cap, scale, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
picked = []
for p in sys.argv[5:5 + k]:
    stages = pickle.load(open(p, 'rb'))
    s = min([s for s in stages if s['n_aux'] <= cap], key=lambda s: s['alpha'])
    picked.append(s)
Ws = [s['W'] / s['g'] for s in picked]
m = max(W.max() for W in Ws)
fam = []
for s, W in zip(picked, Ws):
    i = s['i']
    sat = sat_assignments(k, i)
    prim = [tuple(x[j] for x in sat) for j in range(3)]
    const = [tuple([c] * len(sat)) for c in range(k)]
    vcols = prim + const + [tuple(c) for c in s['cols']]
    Wi = np.rint(W / m * scale).astype(np.int64)
    keep = [u for u in range(Wi.shape[0]) if u < 3 + k or Wi[u].sum() > 0]
    Wi = Wi[np.ix_(keep, keep)]
    kc = [vcols[u] for u in keep]
    idx = {c: j for j, c in enumerate(kc)}
    sidx = {x: j for j, x in enumerate(sat)}
    autos = []
    for pp in PERMS:
        cp = [sidx[act_x(inv(pp), x)] for x in sat]
        autos.append([idx[tuple(c[t] for t in cp)] + 1 for c in kc])
    n = Wi.shape[0]
    fam.append({'i': i, 'n_vertices': n, 'lp_alpha': s['alpha'], 'automorphisms': autos, 'columns': [list(c) for c in kc],
                'edges': [(u + 1, v + 1, int(Wi[u, v])) for u in range(n) for v in range(u + 1, n) if Wi[u, v]]})
    print(f'i={i}: lp alpha={s["alpha"]:.3f} n_aux={n-3-k} edges={len(fam[-1]["edges"])}')
fam.sort(key=lambda g: g['i'])
json.dump({'k': k, 'gadgets': fam}, open(out, 'w'))
print('saved', out, ' LP-predicted A.4 ratio =', 1 - (k - 1) / sum(s['alpha'] for s in picked))
