"""Dynamic single-predicate gadget LP with dual-priced column generation.

LP for the i-gadget (see colsearch.py):   max g
    (comp)  A_c w >= 1           one row per x in C_i (canonical witness coloring)
    (s1)    A   w <= 1           soundness rows with x in C_i
    (s2)    A   w + g <= 1       soundness rows with x in S_i
Pricing a new auxiliary vertex v joined to existing vertices u with weights d_u >= 0:
its canonical value at a satisfying x is a best response
    f(x) in argmax_c sum_u d_u [c != f_u(x)],
and in every soundness row r the adversary also best-responds, so the reduced cost is
    RC(d) = sum_x pi_x max_c sum_u d_u [c != f_u(x)]  -  sum_r y_r max_c sum_u d_u [c != col_r(u)]
(pi, y = duals). RC(d) > 0 means adding v improves the LP to first order.
"""
import time
from itertools import product
import numpy as np
import highspy

from .core import allowed_global_assignments, sat_assignments, FixedMaxCut
from .search import batched_local_search, cpsat_max

INF = highspy.kHighsInf


class DynLP:
    def __init__(self, k, i, columns=(), seed=0, excess=0.0):
        self.k, self.i = k, i
        self.rng = np.random.default_rng(seed)
        self.sat = sat_assignments(k, i)
        self.S = len(self.sat)
        self.Y = allowed_global_assignments(k)
        self.X = list(product(range(k), repeat=3))
        self.items = [(x, y) for x in self.X for y in self.Y]
        self.fixed = np.array([list(x) + list(y) for x, y in self.items], dtype=np.int64)
        self.is_sat = np.array([sum(x) % k == i for x, _ in self.items])
        self.excess = excess
        prim = [tuple(x[j] for x in self.sat) for j in range(3)]
        const = [tuple([c] * self.S) for c in range(k)]
        self.vcols = prim + const               # columns of all vertices
        self.alive = [True] * len(self.vcols)
        self.nf = 3 + k
        self.h = highspy.Highs()
        self.h.setOptionValue('output_flag', False)
        self.h.setOptionValue('presolve', 'off')
        # variable 0 = g (free, objective -1 since HiGHS minimizes)
        self.h.addVar(-INF, INF)
        self.h.changeColCost(0, -1.0)
        self.edge_var = {}                      # (u,v) u<v -> var index
        self.var_edge = [None]
        # completeness rows 0..S-1 (no variables yet)
        for _ in range(self.S):
            self.h.addRow(1.0, INF, 0, np.array([], dtype=np.int32), np.array([]))
        self.row_col = []                       # soundness rows: list of colorings (lists)
        self.row_sat = []
        self.row_key = set()
        self.sol = None
        # edges among fixed vertices
        for u in range(self.nf):
            for v in range(u + 1, self.nf):
                self._add_edge_var(u, v)
        for f in columns:
            self.add_vertex(tuple(int(t) for t in f))

    # ---------------------------------------------------------------- structure
    @property
    def n(self):
        return len(self.vcols)

    def _add_edge_var(self, u, v):
        """Add LP variable for edge (u,v); coefficients in all existing rows."""
        fu, fv = np.array(self.vcols[u]), np.array(self.vcols[v])
        rows, vals = [], []
        comp = (fu != fv)
        for r in np.nonzero(comp)[0]:
            rows.append(r)
            vals.append(1.0)
        if self.row_col:
            C = self._colmat()
            cut = C[:, u] != C[:, v]
            for r in np.nonzero(cut)[0]:
                rows.append(self.S + r)
                vals.append(1.0)
        idx = len(self.var_edge)
        self.h.addCol(0.0, 0.0, INF, len(rows), np.array(rows, dtype=np.int32), np.array(vals))
        self.edge_var[(u, v)] = idx
        self.var_edge.append((u, v))

    def _colmat(self):
        if getattr(self, '_C', None) is None or self._C.shape != (len(self.row_col), self.n):
            self._C = np.array(self.row_col, dtype=np.int64).reshape(len(self.row_col), self.n)
        return self._C

    def add_vertex(self, f, fill=None):
        """Add aux vertex with column f.  fill[r] = its color in existing soundness row r."""
        f = tuple(int(t) for t in f)
        if f in self.vcols:
            return None
        v = self.n
        if fill is None:
            fill = self.rng.integers(0, self.k, size=len(self.row_col))
        for r, c in enumerate(fill):
            self.row_col[r].append(int(c))
        self.vcols.append(f)
        self.alive.append(True)
        self._C = None
        for u in range(v):
            if self.alive[u]:
                self._add_edge_var(u, v)
        return v

    def kill_vertex(self, v):
        """Fix all edge weights of v to zero (vertex stays in the index space)."""
        self.alive[v] = False
        for (a, b), j in self.edge_var.items():
            if a == v or b == v:
                self.h.changeColBounds(j, 0.0, 0.0)

    def add_sound_row(self, col, sat):
        key = tuple(int(c) for c in col)
        if key in self.row_key:
            return False
        self.row_key.add(key)
        idx, vals = [], []
        for (u, v), j in self.edge_var.items():
            if key[u] != key[v]:
                idx.append(j)
                vals.append(1.0)
        if not sat:
            idx.append(0)
            vals.append(1.0)
        self.h.addRow(-INF, 1.0 + (self.excess if sat else 0.0), len(idx),
                      np.array(idx, dtype=np.int32), np.array(vals))
        self.row_col.append(list(key))
        self.row_sat.append(bool(sat))
        self._C = None
        return True

    # ---------------------------------------------------------------- solve
    def solve(self):
        self.h.run()
        st = self.h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(str(st))
        s = self.h.getSolution()
        self.sol = np.array(s.col_value)
        self.dual = np.array(s.row_dual)
        return self.sol[0]

    def W(self):
        n = self.n
        W = np.zeros((n, n))
        for (u, v), j in self.edge_var.items():
            w = max(self.sol[j], 0.0)
            W[u, v] = W[v, u] = w
        return W

    def bounds(self):
        g = self.sol[0]
        return np.where(self.is_sat, 1.0 + self.excess, 1.0 - g)

    def separate(self, restarts=8, rounds=3, tol=1e-7, max_add=None):
        W = self.W()
        vals, cols = batched_local_search(W, self.k, self.fixed, self.nf,
                                          restarts=restarts, rounds=rounds, rng=self.rng)
        viol = vals - self.bounds()
        order = np.argsort(-viol)
        added = 0
        for j in order:
            if viol[j] <= tol or (max_add and added >= max_add):
                break
            if self.add_sound_row(cols[j], self.is_sat[j]):
                added += 1
        return added, float(max(viol.max(), 0.0))

    def seed_random(self):
        for j in range(len(self.items)):
            col = np.concatenate([self.fixed[j], self.rng.integers(0, self.k, size=self.n - self.nf)])
            self.add_sound_row(col, self.is_sat[j])

    def optimize(self, max_iter=200, restarts=8, rounds=3, verbose=False):
        if not self.row_col:
            self.seed_random()
        for it in range(max_iter):
            g = self.solve()
            added, worst = self.separate(restarts=restarts, rounds=rounds)
            if verbose:
                print(f'    it {it}: g={g:.7f} added={added} worst={worst:.2g} rows={len(self.row_col)}', flush=True)
            if added == 0:
                break
        return self.solve()

    # ---------------------------------------------------------------- pricing
    def duals(self):
        """pi (completeness, >= 0) and y (soundness, >= 0) in the sign convention of
        the docstring (checked by reduced_cost_check)."""
        d = self.dual
        pi = np.abs(d[:self.S])
        y = np.abs(d[self.S:])
        return pi, y

    def rc_existing(self):
        """Reduced costs of existing edge variables under the docstring's formula."""
        pi, y = self.duals()
        C = self._colmat()
        V = np.array(self.vcols)        # n x S
        out = {}
        for (u, v), j in self.edge_var.items():
            comp = (V[u] != V[v]).astype(float) @ pi
            snd = (C[:, u] != C[:, v]).astype(float) @ y
            out[(u, v)] = comp - snd
        return out

    def price(self, N, d, rng=None, act_rows=None):
        """Return (RC, f, fill) for a new vertex adjacent to vertices N with weights d."""
        rng = self.rng if rng is None else rng
        k = self.k
        pi, y = self.duals()
        V = np.array(self.vcols)[N]           # |N| x S
        d = np.asarray(d, dtype=float)
        D = d.sum()
        # completeness: per sat point, the gain of color c = D - sum_u d_u [f_u(x) = c]
        same = np.stack([(d[:, None] * (V == c)).sum(0) for c in range(k)], 1)   # S x k
        best = same.min(1)
        f = []
        for s in range(self.S):
            cands = np.nonzero(same[s] <= best[s] + 1e-12)[0]
            f.append(int(rng.choice(cands)))
        comp = ((D - best) * pi).sum()
        C = self._colmat()
        rows = np.nonzero(y > 1e-12)[0] if act_rows is None else act_rows
        Cr = C[rows][:, N]                     # R x |N|
        sameR = np.stack([((Cr == c) * d[None, :]).sum(1) for c in range(k)], 1)
        snd = ((D - sameR.min(1)) * y[rows]).sum()
        # fill colors for all rows: adversary best response
        CA = C[:, N]
        sameA = np.stack([((CA == c) * d[None, :]).sum(1) for c in range(k)], 1)
        fill = sameA.argmin(1)
        return comp - snd, tuple(f), fill

    def used_vertices(self, tol=1e-9):
        W = self.W()
        return [u for u in range(self.nf, self.n) if self.alive[u] and W[u].sum() > tol]


def generate(lp, n_samples=3000, max_support=5, top=4, hill=30, rng=None, wmax=3):
    """Sample sparse directions d, hill-climb the best ones, return top new columns."""
    rng = lp.rng if rng is None else rng
    alive = [u for u in range(lp.n) if lp.alive[u]]
    W = lp.W()
    deg = W.sum(1)
    pi, y = lp.duals()
    act = np.nonzero(y > 1e-12)[0]
    cand = []
    for _ in range(n_samples):
        s = int(rng.integers(2, max_support + 1))
        # prefer vertices with weight
        p = np.array([deg[u] + 0.05 * deg.max() + 1e-9 for u in alive])
        N = list(rng.choice(alive, size=min(s, len(alive)), replace=False, p=p / p.sum()))
        d = list(rng.integers(1, wmax + 1, size=len(N)))
        rc, f, _ = lp.price(N, d, rng=rng, act_rows=act)
        cand.append((rc / sum(d), N, d))
    cand.sort(key=lambda t: -t[0])
    out = []
    seen = set(lp.vcols)
    for rc0, N, d in cand[: top * 3]:
        # hill climb on (N, d)
        best = (rc0, N, d)
        for _ in range(hill):
            N2, d2 = list(best[1]), list(best[2])
            r = rng.random()
            if r < 0.3 and len(N2) < 8:
                u = int(rng.choice(alive))
                if u not in N2:
                    N2.append(u)
                    d2.append(int(rng.integers(1, wmax + 1)))
            elif r < 0.5 and len(N2) > 2:
                j = int(rng.integers(len(N2)))
                N2.pop(j)
                d2.pop(j)
            else:
                j = int(rng.integers(len(N2)))
                d2[j] = max(1, d2[j] + int(rng.choice([-1, 1])))
            rc, f, _ = lp.price(N2, d2, rng=rng, act_rows=act)
            rc /= sum(d2)
            if rc > best[0]:
                best = (rc, N2, d2)
        rc, N, d = best
        if rc <= 1e-9:
            continue
        # a few tie-breaking variants
        for _ in range(3):
            rcv, f, fill = lp.price(N, d, rng=rng)
            if f not in seen:
                seen.add(f)
                out.append((rc, f, fill, N, d))
                break
        if len(out) >= top:
            break
    return out
