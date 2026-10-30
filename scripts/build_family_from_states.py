"""Build an integer gadget family JSON (with S_3 automorphism certificates and columns) from
converged per-predicate LP states (dicts with vcols, eu, ev, eorb, orbit_w, g_*).
Each gadget is scaled to unit gap (weights / g_i), then all are rounded with a common scale.
Usage: build_family_from_states.py k scale out.json state_0 ... state_{k-1}"""
import sys
import os, json, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.core import sat_assignments
from gadgets.symlp import PERMS, act_x, inv

k, scale, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
states = [pickle.load(open(p, 'rb')) for p in sys.argv[4:4 + k]]
Ws, metas = [], []
for st in states:
    i = st['i']
    vcols = [tuple(c) for c in st['vcols']]
    n = len(vcols)
    W = np.zeros((n, n))
    w = np.maximum(np.array(st['orbit_w'], dtype=float), 0.0)
    eu, ev, eo = np.array(st['eu']), np.array(st['ev']), np.array(st['eorb'])
    W[eu, ev] = w[eo]
    W[ev, eu] = w[eo]
    g = st.get('g_strong', st.get('g_std'))
    Ws.append(W / g)
    metas.append((i, vcols, g))
m = max(W.max() for W in Ws)
fam = []
for W, (i, vcols, g) in zip(Ws, metas):
    Wi = np.rint(W / m * scale).astype(np.int64)
    keep = [u for u in range(Wi.shape[0]) if u < 3 + k or Wi[u].sum() > 0]
    Wi = Wi[np.ix_(keep, keep)]
    kc = [vcols[u] for u in keep]
    idx = {c: j for j, c in enumerate(kc)}
    sat = sat_assignments(k, i)
    sidx = {x: j for j, x in enumerate(sat)}
    autos = []
    for p in PERMS:
        cp = [sidx[act_x(inv(p), x)] for x in sat]
        autos.append([idx[tuple(c[t] for t in cp)] + 1 for c in kc])
    n = Wi.shape[0]
    fam.append({'i': i, 'n_vertices': n, 'lp_alpha': 1 / g, 'automorphisms': autos,
                'columns': [list(c) for c in kc],
                'edges': [(u + 1, v + 1, int(Wi[u, v])) for u in range(n) for v in range(u + 1, n) if Wi[u, v]]})
    print(f'i={i}: lp alpha={1/g:.3f} n_aux={n-3-k} edges={len(fam[-1]["edges"])}', flush=True)
fam.sort(key=lambda g: g['i'])
json.dump({'k': k, 'gadgets': fam}, open(out, 'w'))
print('saved', out, ' LP-predicted A.4 ratio =', 1 - (k - 1) / sum(1 / mm[2] for mm in metas))
