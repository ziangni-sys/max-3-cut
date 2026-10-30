"""Persistent-LP symmetric column generation for one i-gadget, with periodic compaction.
Usage: run_search2.py k i seed init(ckpt|pkl|none) [max_aux] [tag]"""
import sys, os, time, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.symlp import SymDynLP, generate_sym

k, i, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
init = sys.argv[4]
max_aux = int(sys.argv[5]) if len(sys.argv) > 5 else 70
tag = sys.argv[6] if len(sys.argv) > 6 else f's{seed}'
ck = f'{_ROOT}/results/s2_k{k}_i{i}_{tag}_cap{os.environ.get("CAP", "28")}.ckpt'
rng = np.random.default_rng(seed)
nf = 3 + k
REL_TOL = 1e-3

def A(g):
    return 1 / g if g > 1e-12 else float('inf')

def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)

def load_init():
    if os.path.exists(ck):
        st = pickle.load(open(ck, 'rb'))
        log(f'resume {ck}: best alpha={A(st["best"]["g"]):.4f}')
        return st['cols'], st['pool'], st['best'], st['hist']
    if init == 'none':
        return [], [], dict(g=-1.0), []
    st = pickle.load(open(init, 'rb'))
    if 'cols' in st and 'best' not in st:
        return list(st['cols']), [], dict(g=-1.0), []
    if 'best' in st:
        return list(st['best']['cols']), st.get('pool', []), dict(g=-1.0), []
    return dict(st['groups'])[i], [], dict(g=-1.0), []

Y0_ROUNDS = int(os.environ.get('Y0_ROUNDS', '0'))   # continuation: first rounds with globals pinned to y0
state = {'y0': Y0_ROUNDS > 0}

def build(cols, pool):
    Ylist = [tuple(range(k))] if state['y0'] else None
    lp = SymDynLP(k, i, cols, seed=int(rng.integers(1 << 30)), Y=Ylist)
    pc, ps = [], []
    for sat, xy, cmap in pool:
        if state['y0'] and tuple(xy[3:]) != tuple(range(k)):
            continue
        pc.append(list(xy) + [cmap.get(f, int(rng.integers(k))) for f in lp.vcols[nf:]])
        ps.append(sat)
    if pc:
        lp.add_sound_rows(pc, ps)
    if not lp.row_col:
        lp.seed_random()
    return lp

def harvest(lp, max_rows=8000):
    act = np.array(lp.h.getSolution().row_value)[lp.S:]
    ub = np.where(np.array(lp.row_sat), 1.0, 1.0)
    slack = ub - act
    used = set(lp.used_vertices())
    pool = []
    for r in np.nonzero(slack < 1e-6)[0]:
        col = lp.row_col[r]
        pool.append((lp.row_sat[r], tuple(col[:nf]), {lp.vcols[u]: col[u] for u in range(nf, len(col)) if u in used}))
    if len(pool) > max_rows:
        pool = [pool[j] for j in rng.choice(len(pool), max_rows, replace=False)]
    return pool

def save(cols, pool, best, hist):
    tmp = ck + '.tmp'
    pickle.dump(dict(cols=cols, pool=pool, best=best, hist=hist, k=k, i=i), open(tmp, 'wb'))
    os.replace(tmp, ck)

CAP = int(os.environ.get('CAP', '28'))
cols, pool, best, hist = load_init()
lp = build(cols, pool)
g = lp.optimize(rel_tol=REL_TOL)
if lp.prune_to(CAP):
    g = lp.optimize(rel_tol=REL_TOL)
    cols = [lp.vcols[u] for u in lp.used_vertices()]
    lp = build(cols, harvest(lp))
    g = lp.optimize(rel_tol=REL_TOL)
log(f'start alpha={A(g):.4f} n_aux={lp.n-nf}')
since = 0
rounds = 0
while True:
    rounds += 1
    if state['y0'] and rounds > Y0_ROUNDS:
        state['y0'] = False
        cols = [lp.vcols[u] for u in lp.used_vertices()]
        pool = harvest(lp)
        lp = build(cols, pool)
        g = lp.optimize(rel_tol=REL_TOL)
        best = dict(g=-1.0)
        log(f'  switched to full Y: alpha={A(g):.4f} n_aux={lp.n-nf}')
    t0 = time.time()
    new = generate_sym(lp, n_samples=1500, top=4)
    for rc, f, fills, N, d in new:
        lp.add_orbit(f, fills=fills)
    g = lp.optimize(rel_tol=REL_TOL)
    if lp.prune_to(CAP):
        g = lp.optimize(rel_tol=REL_TOL)
    used = lp.used_vertices()
    W = lp.W()
    if g > best['g'] + 1e-10 and not state['y0']:
        keep = list(range(nf)) + used
        best = dict(g=float(g), cols=[lp.vcols[u] for u in used], W=W[np.ix_(keep, keep)].copy())
    since += 1
    hist.append((float(g), len(used), lp.n - nf, len(lp.row_col), time.time() - t0))
    log(f'{"[y0] " if state["y0"] else ""}alpha={A(g):.4f} used={len(used)} n_aux={lp.n-nf} rows={len(lp.row_col)} added_orbits={len(new)} best={A(best["g"]):.4f} {time.time()-t0:.0f}s')
    if lp.n - nf > max_aux or len(lp.row_col) > 20000 or since >= 10:
        pool = harvest(lp)
        cols = [lp.vcols[u] for u in used]
        lp = build(cols, pool)
        g = lp.optimize(rel_tol=REL_TOL)
        since = 0
        log(f'  compacted: n_aux={lp.n-nf} rows={len(lp.row_col)} alpha={A(g):.4f}')
    save([lp.vcols[u] for u in lp.used_vertices()] if lp.sol is not None else cols, pool, best, hist)
