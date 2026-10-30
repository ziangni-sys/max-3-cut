"""Restartable column-generation search for a single i-gadget (k colors).

Outer iteration: rebuild DynLP on the current used columns + warm-start rows (pool),
optimize with heuristic separation, run a few rounds of dual-priced column
generation, then prune unused vertices, harvest tight rows, checkpoint.
"""
import os
import pickle
import time
import numpy as np

from .colgen import DynLP, generate
from .core import FixedMaxCut, allowed_global_assignments, sat_assignments
from .cpsat import cpsat_max


class Search:
    def __init__(self, k, i, init_cols, seed=0, ckpt=None, max_cols=40, log=print):
        self.k, self.i = k, i
        self.nf = 3 + k
        self.rng = np.random.default_rng(seed)
        self.ckpt = ckpt
        self.max_cols = max_cols
        self.log = log
        self.cols = [tuple(int(v) for v in c) for c in init_cols]
        self.age = {}             # column -> consecutive outer iterations without weight
        self.age_max = 3
        self.pool = []
        self.best = dict(g=-1.0, cols=None, w=None, verified=False)
        self.history = []
        if ckpt and os.path.exists(ckpt):
            st = pickle.load(open(ckpt, 'rb'))
            self.cols, self.pool, self.best, self.history = st['cols'], st['pool'], st['best'], st['history']
            self.age = st.get('age', {})
            self.rng = np.random.default_rng(seed + len(self.history) + 1)
            log(f'resumed from {ckpt}: best g={self.best["g"]:.7f} alpha={1/self.best["g"]:.4f} cols={len(self.cols)} pool={len(self.pool)}')

    def save(self):
        if self.ckpt:
            tmp = self.ckpt + '.tmp'
            pickle.dump(dict(cols=self.cols, pool=self.pool, best=self.best, history=self.history, age=self.age,
                             k=self.k, i=self.i), open(tmp, 'wb'))
            os.replace(tmp, self.ckpt)

    def build(self, cols):
        lp = DynLP(self.k, self.i, cols, seed=int(self.rng.integers(1 << 30)))
        for sat, xy, cmap in self.pool:
            col = list(xy) + [cmap.get(f, int(self.rng.integers(self.k))) for f in lp.vcols[self.nf:]]
            lp.add_sound_row(col, sat)
        if not lp.row_col:
            lp.seed_random()
        return lp

    def harvest(self, lp, max_rows=12000):
        act = np.array(lp.h.getSolution().row_value)[lp.S:]
        ub = np.where(np.array(lp.row_sat), 1.0 + lp.excess, 1.0)
        slack = ub - act
        used = set(u for u in range(self.nf, lp.n) if lp.alive[u])
        pool = []
        for r in np.nonzero(slack < 1e-6)[0]:
            col = lp.row_col[r]
            pool.append((lp.row_sat[r], tuple(col[:self.nf]),
                         {lp.vcols[u]: col[u] for u in range(self.nf, len(col)) if u in used}))
        if len(pool) > max_rows:
            idx = self.rng.choice(len(pool), max_rows, replace=False)
            pool = [pool[j] for j in idx]
        self.pool = pool

    def outer(self, inner_rounds=3, n_samples=2000, top=4, restarts=8):
        t0 = time.time()
        lp = self.build(self.cols)
        g = lp.optimize(restarts=restarts)
        t1 = time.time()
        for r in range(inner_rounds):
            new = generate(lp, n_samples=n_samples, top=top)
            for rc, f, fill, N, d in new:
                lp.add_vertex(f, fill=fill)
            g = lp.optimize(restarts=restarts)
        t2 = time.time()
        used = lp.used_vertices()
        W = lp.W()
        wsum = W.sum(1)
        used.sort(key=lambda u: -wsum[u])
        uset = set(used)
        # keep recently added unused candidates for a few iterations (they can become
        # useful in combination with later candidates)
        keepers = []
        for u in range(self.nf, lp.n):
            f = lp.vcols[u]
            self.age[f] = 0 if u in uset else self.age.get(f, 0) + 1
            if u in uset or self.age[f] < self.age_max:
                keepers.append(u)
        keepers.sort(key=lambda u: (self.age[lp.vcols[u]], -wsum[u]))
        dropped = max(0, sum(1 for u in keepers[:self.max_cols] if u in uset) < len(used))
        keepers = keepers[:self.max_cols]
        self.cols = [lp.vcols[u] for u in keepers]
        self.harvest(lp)
        if g > self.best['g'] + 1e-10:
            keep = list(range(self.nf)) + used
            self.best = dict(g=float(g), cols=list(self.cols), W=W[np.ix_(keep, keep)].copy(), verified=False)
        self.history.append((float(g), len(used), dropped, len(lp.row_col), time.time() - t0))
        self.save()
        self.log(f'g={g:.7f} alpha={1/g:.4f} used={len(used)} dropped={dropped} rows={len(lp.row_col)} '
                 f'best_alpha={1/self.best["g"]:.4f} t_opt={t1-t0:.0f}s t_gen={t2-t1:.0f}s')
        return g


def exact_check_W(W, k, i, tol=1e-9, scale=10 ** 6, method='auto', workers=1):
    """Exact (up to integer scaling for CP-SAT) C, S1, S2 of a real-weighted gadget W.
    Returns params and the list of (value, sat, coloring) for every (x,y)."""
    from itertools import product
    n = W.shape[0]
    nf = 3 + k
    used = [u for u in range(nf, n) if W[u].sum() > tol]
    Y = allowed_global_assignments(k)
    y0 = tuple(range(k))
    if method == 'auto':
        method = 'mitm' if k ** ((len(used) + 1) // 2) <= 5000 else 'cpsat'
    if method == 'mitm':
        fm = FixedMaxCut(W, k, aux=used, integer=False)
        def mx(x, y):
            v, z = fm.maxval(x, y, return_arg=True)
            return v, list(x) + list(y) + [z.get(u, 0) for u in range(nf, n)]
    else:
        m = W.max()
        Wi = np.rint(W / m * scale).astype(np.int64)
        def mx(x, y):
            v, col = cpsat_max(Wi, k, list(x) + list(y), nf, workers=workers)
            col = np.array(col)
            colors = col
            val = float((W * (colors[:, None] != colors[None, :])).sum() / 2)
            return val, list(colors)
    sat = [x for x in product(range(k), repeat=3) if sum(x) % k == i]
    C = min(mx(x, y0)[0] for x in sat)
    out = []
    S1 = S2 = -np.inf
    for x in product(range(k), repeat=3):
        for y in Y:
            v, col = mx(x, y)
            s = sum(x) % k == i
            out.append((v, s, col))
            if s:
                S1 = max(S1, v)
            else:
                S2 = max(S2, v)
    return dict(C=C, S1=S1, S2=S2), out


class SymSearch(Search):
    """Same outer loop on the S_3-symmetric LP; candidates are added as orbits."""

    def build(self, cols):
        from .symlp import SymDynLP
        lp = SymDynLP(self.k, self.i, cols, seed=int(self.rng.integers(1 << 30)))
        for sat, xy, cmap in self.pool:
            col = list(xy) + [cmap.get(f, int(self.rng.integers(self.k))) for f in lp.vcols[self.nf:]]
            lp.add_sound_row(col, sat)
        if not lp.row_col:
            lp.seed_random()
        return lp

    def outer(self, inner_rounds=3, n_samples=1500, top=3, restarts=8, rel_tol=1e-3):
        from .symlp import generate_sym
        t0 = time.time()
        lp = self.build(self.cols)
        g = lp.optimize(restarts=restarts, rel_tol=rel_tol)
        t1 = time.time()
        for r in range(inner_rounds):
            new = generate_sym(lp, n_samples=n_samples, top=top)
            for rc, f, fills, N, d in new:
                lp.add_orbit(f, fills=fills)
            g = lp.optimize(restarts=restarts, rel_tol=rel_tol)
        t2 = time.time()
        used = lp.used_vertices()
        W = lp.W()
        wsum = W.sum(1)
        uset = set(used)
        keepers = []
        for u in range(self.nf, lp.n):
            f = lp.vcols[u]
            self.age[f] = 0 if u in uset else self.age.get(f, 0) + 1
            if u in uset or self.age[f] < self.age_max:
                keepers.append(u)
        # drop whole orbits only: keep a column iff its orbit representative is kept
        keepers.sort(key=lambda u: (self.age[lp.vcols[u]], -wsum[u]))
        kept = []
        for u in keepers:
            if len(kept) >= self.max_cols:
                break
            kept.append(lp.vcols[u])
        kset = set(kept)
        for f in list(kept):
            for gcol in lp.orbit_cols(f):
                kset.add(gcol)
        self.cols = sorted(kset)
        self.harvest(lp)
        if g > self.best['g'] + 1e-10:
            keep = list(range(self.nf)) + used
            self.best = dict(g=float(g), cols=[lp.vcols[u] for u in used], W=W[np.ix_(keep, keep)].copy(), verified=False)
        self.history.append((float(g), len(used), 0, len(lp.row_col), time.time() - t0))
        self.save()
        self.log(f'g={g:.7f} alpha={1/g:.4f} used={len(used)} cand={lp.n - self.nf} rows={len(lp.row_col)} '
                 f'best_alpha={1/self.best["g"]:.4f} t_opt={t1-t0:.0f}s t_gen={t2-t1:.0f}s')
        return g
