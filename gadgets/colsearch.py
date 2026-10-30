"""Per-predicate gadget optimization + local search over the set of auxiliary columns.

For a fixed predicate i the joint hardness ratio of Thm A.4 only depends on the
gadgets through alpha_i = C_i / (C_i - S2_i) when S1_i = C_i:
    ratio = 1 - (k-1) / sum_i alpha_i         (scale gadget i so all gaps are equal).
So each i-gadget can be optimized separately:
    max g  s.t.  cut(w; canonical coloring at x) >= 1        x in C_i
                 cut(w; col) <= 1                            soundness rows, x in C_i
                 cut(w; col) + g <= 1                        soundness rows, x in S_i
                 w >= 0
alpha_i = 1/g.  Columns (aux vertices) are functions C_i -> Z_k; the outer loop
proposes new columns (mutations / crossovers of used columns, random columns),
re-optimizes, and prunes columns that carry no weight.
"""
import time
from itertools import product
import numpy as np
import highspy

from .core import allowed_global_assignments, sat_assignments, FixedMaxCut
from .search import Group, batched_local_search, cpsat_max


class SingleLP:
    def __init__(self, k, i, columns, seed=0, excess=0.0):
        self.k, self.i = k, i
        self.g = Group(k, i, columns)
        self.rng = np.random.default_rng(seed)
        self.Y = allowed_global_assignments(k)
        self.X = list(product(range(k), repeat=3))
        self.items = [(x, y) for x in self.X for y in self.Y]
        self.fixed = np.array([list(x) + list(y) for x, y in self.items], dtype=np.int64)
        self.is_sat = np.array([sum(x) % k == i for x, _ in self.items])
        E = self.g.E
        self.gidx = E
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        inf = highspy.kHighsInf
        h.addVars(E + 1, np.zeros(E + 1), np.full(E + 1, inf))
        cost = np.zeros(E + 1)
        cost[self.gidx] = -1.0                      # maximize g
        h.changeColsCost(E + 1, np.arange(E + 1, dtype=np.int32), cost)
        h.changeColBounds(self.gidx, -inf, inf)
        self.h = h
        self.excess = excess
        for row in self.g.cut_coeffs(self.g.completeness_colorings()):
            nz = np.nonzero(row)[0]
            self._row(nz, row[nz], 1.0, inf)
        self.rows = {}           # key -> (row index)
        self.nrows = len(self.g.sat)
        self.sol = None

    def _row(self, idx, val, lo, hi):
        self.h.addRow(lo, hi, len(idx), np.asarray(idx, dtype=np.int32), np.asarray(val, dtype=np.float64))

    def add_sound(self, sat, col):
        key = tuple(int(c) for c in col)
        if key in self.rows:
            return False
        coef = self.g.cut_coeffs(np.asarray(col))
        nz = np.nonzero(coef)[0]
        inf = highspy.kHighsInf
        if sat:
            self._row(nz, coef[nz], -inf, 1.0 + self.excess)
        else:
            self._row(np.append(nz, self.gidx), np.append(coef[nz], 1.0), -inf, 1.0)
        self.rows[key] = self.nrows
        self.nrows += 1
        return True

    def solve(self):
        self.h.run()
        st = self.h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(str(st))
        self.sol = np.array(self.h.getSolution().col_value)
        return self.sol[self.gidx]

    def w(self):
        return np.maximum(self.sol[:self.g.E], 0.0)

    def W(self):
        return self.g.W_from_w(self.w())

    def bounds(self):
        gval = self.sol[self.gidx]
        return np.where(self.is_sat, 1.0 + self.excess, 1.0 - gval)

    def separate_heuristic(self, restarts=8, rounds=3, tol=1e-7, max_add=None):
        W = self.W()
        vals, cols = batched_local_search(W, self.k, self.fixed, 3 + self.k,
                                          restarts=restarts, rounds=rounds, rng=self.rng)
        viol = vals - self.bounds()
        order = np.argsort(-viol)
        added = 0
        for j in order:
            if viol[j] <= tol or (max_add and added >= max_add):
                break
            if self.add_sound(self.is_sat[j], cols[j]):
                added += 1
        return added, float(max(viol.max(), 0))

    def seed_random(self, per_item=1):
        n = self.g.n
        for j in range(len(self.items)):
            for _ in range(per_item):
                col = np.concatenate([self.fixed[j], self.rng.integers(0, self.k, size=n - 3 - self.k)])
                self.add_sound(self.is_sat[j], col)

    def optimize(self, max_iter=300, restarts=8, rounds=3, verbose=False):
        it = 0
        while True:
            gval = self.solve()
            added, worst = self.separate_heuristic(restarts=restarts, rounds=rounds)
            if verbose:
                print(f'    it {it} g={gval:.6f} alpha={1/max(gval,1e-12):.4f} added={added} worst={worst:.2g} rows={self.nrows}', flush=True)
            it += 1
            if added == 0 or it >= max_iter:
                break
        return self.solve()

    def column_weight(self):
        """Total incident weight of every aux column."""
        W = self.W()
        nf = 3 + self.k
        return W[nf:].sum(1)

    def tight_rows(self):
        """Return colorings of soundness rows that are (nearly) tight at the optimum."""
        sol = self.h.getSolution()
        act = np.array(sol.row_value)
        out = []
        b = self.bounds_rowwise()
        return out


def exact_check(k, i, columns, w, tol=1e-9, use_cpsat=False, scale=None):
    """Exact (float) max over aux for all (x, y) of the single-gadget LP solution;
    returns (C, S1, S2) of the *real-weighted* gadget, using MITM on used vertices."""
    g = Group(k, i, columns)
    W = g.W_from_w(w)
    n = W.shape[0]
    used = [u for u in range(3 + k, n) if W[u].sum() > tol]
    fm = FixedMaxCut(W, k, aux=used, integer=False)
    Y = allowed_global_assignments(k)
    sat = sat_assignments(k, i)
    y0 = tuple(range(k))
    C = min(fm.maxval(x, y0) for x in sat)
    S1 = S2 = -1
    worst = []
    for x in product(range(k), repeat=3):
        for y in Y:
            v, z = fm.maxval(x, y, return_arg=True)
            col = list(x) + list(y) + [z.get(u, 0) for u in range(3 + k, n)]
            if sum(x) % k == i:
                S1 = max(S1, v)
            else:
                S2 = max(S2, v)
            worst.append((v, sum(x) % k == i, col))
    return C, S1, S2, worst


class ColumnSearch:
    def __init__(self, k, i, init_columns, seed=0, max_cols=20, log=print):
        self.k, self.i = k, i
        self.rng = np.random.default_rng(seed)
        self.sat = sat_assignments(k, i)
        self.cols = [tuple(c) for c in init_columns]
        self.max_cols = max_cols
        self.pool = []          # (sat, {column: color}, x, y) adversarial rows to warm start
        self.log = log
        self.best = (-1, None, None)

    def _lp(self, cols, seed):
        lp = SingleLP(self.k, self.i, cols, seed=seed)
        # warm start rows from the pool, filling unknown columns with random colors
        vcols = lp.g.vcols
        nf = 3 + self.k
        for sat, cmap, xy in self.pool:
            col = list(xy) + [cmap.get(f, int(self.rng.integers(self.k))) for f in vcols[nf:]]
            lp.add_sound(sat, col)
        if not self.pool:
            lp.seed_random()
        return lp

    def _harvest(self, lp, max_rows=20000):
        """Keep the rows that are tight at the optimum as the warm-start pool."""
        sol = lp.h.getSolution()
        act = np.array(sol.row_value)
        vcols = lp.g.vcols
        nf = 3 + self.k
        gval = lp.sol[lp.gidx]
        pool = []
        for key, r in lp.rows.items():
            col = key
            x = tuple(col[:3])
            sat = sum(x) % self.k == self.i
            # row activity: coef.w (+ g for unsat rows)
            slack = (1.0 + lp.excess - act[r]) if sat else (1.0 - act[r])
            if slack < 1e-6:
                pool.append((sat, {vcols[u]: col[u] for u in range(nf, len(col))}, col[:nf]))
        if len(pool) > max_rows:
            idx = self.rng.choice(len(pool), max_rows, replace=False)
            pool = [pool[j] for j in idx]
        self.pool = pool

    def propose(self, used, nprop):
        k = self.k
        m = len(self.sat)
        props = []
        for _ in range(nprop):
            r = self.rng.random()
            if used and r < 0.55:
                f = list(used[self.rng.integers(len(used))])
                for _ in range(1 + int(self.rng.random() < 0.4)):
                    p = self.rng.integers(m)
                    f[p] = (f[p] + 1 + self.rng.integers(k - 1)) % k
            elif len(used) >= 2 and r < 0.8:
                a, b = self.rng.choice(len(used), 2, replace=False)
                mask = self.rng.random(m) < 0.5
                f = [used[a][j] if mask[j] else used[b][j] for j in range(m)]
            elif used and r < 0.9:
                # relabel colors of a used column by a random permutation
                f0 = used[self.rng.integers(len(used))]
                perm = self.rng.permutation(k)
                f = [int(perm[v]) for v in f0]
            else:
                f = list(self.rng.integers(0, k, size=m))
            props.append(tuple(int(v) for v in f))
        return props

    def step(self, nprop=6, restarts=8):
        cand = list(dict.fromkeys(self.cols + self.propose(self.cols, nprop)))
        lp = self._lp(cand, seed=int(self.rng.integers(1 << 30)))
        gval = lp.optimize(restarts=restarts)
        cw = lp.column_weight()
        cols = lp.g.columns
        used = [(cw[j], cols[j]) for j in range(len(cols)) if cw[j] > 1e-7]
        used.sort(key=lambda t: -t[0])
        dropped = 0
        if len(used) > self.max_cols:
            dropped = len(used) - self.max_cols
            used = used[:self.max_cols]
        self._harvest(lp)
        self.cols = [c for _, c in used]
        if gval > self.best[0] and dropped == 0:
            self.best = (gval, list(cols), lp.w().copy())
        return gval, len(self.cols), dropped, lp.nrows
