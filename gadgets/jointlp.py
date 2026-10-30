"""Joint LP over all k gadgets (each S_3-symmetric) for the shared-global refinement.

Refined soundness (all copies share the global assignment y, normalized into Y):
    b' * sum_i t_i = max_{y in Y} [ sum_i S2_i(y) + max(0, max_r S1_r(y) - S2_r(y)) ]
with S1_i(y) = max_{x in C_i, z} I_i(x,y,z), S2_i(y) = max_{x in S_i, z} I_i(x,y,z).
(refined=False collapses the per-y variables and gives NRT's Thm A.4.)

LP:  min tau
     cut_i(w_i; canonical at x) - c_i >= 0          x in C_i
     sum_i c_i >= 1
     tau - sum_i s2[i,y] - s1[r,y] + s2[r,y] >= 0     all y, r
     tau - sum_i s2[i,y] >= 0                         all y
     s1[i,y] (x in C_i) / s2[i,y] (x in S_i)  >=  cut_i(w_i; row coloring)   soundness rows
"""
from itertools import product
import numpy as np
import highspy

from .core import allowed_global_assignments, sat_assignments
from .search import batched_local_search
from .symlp import PERMS, act_x, inv

INF = highspy.kHighsInf


class Block:
    def __init__(self, P, i):
        k = P.k
        self.P, self.i, self.k = P, i, k
        self.sat = sat_assignments(k, i)
        self.S = len(self.sat)
        self.sidx = {x: j for j, x in enumerate(self.sat)}
        self.colperm = [np.array([self.sidx[act_x(inv(p), x)] for x in self.sat]) for p in PERMS]
        reps = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3)))
        self.items = [(x, y) for x in reps for y in P.Y]
        self.fixed = np.array([list(x) + list(y) for x, y in self.items], dtype=np.int64)
        self.is_sat = np.array([sum(x) % k == i for x, _ in self.items])
        self.svar = np.array([P.s_idx(i, y, 1 if s else 2) for (x, y), s in zip(self.items, self.is_sat)])
        prim = [tuple(x[j] for x in self.sat) for j in range(3)]
        const = [tuple([c] * self.S) for c in range(k)]
        self.vcols = prim + const
        self.vindex = {f: u for u, f in enumerate(self.vcols)}
        self.alive = [True] * len(self.vcols)
        self.rng = P.rng
        self.nf = 3 + k
        self.eu, self.ev, self.eorb = [], [], []
        self.edge_set = {}
        self.ovar = []                  # orbit -> HiGHS column
        self.comp_rows = []
        for _ in range(self.S):         # cut - c_i >= 0
            self.comp_rows.append(P.add_row([P.c_idx(i)], [-1.0], 0.0, INF))
        self.rows, self.row_col, self.row_sat, self.row_key = [], [], [], set()
        self._C = None
        self._new_edges(range(len(self.vcols)))

    @property
    def n(self):
        return len(self.vcols)

    def pcol(self, p_idx, f):
        return tuple(int(v) for v in np.asarray(f)[self.colperm[p_idx]])

    def vperm(self, p_idx, u):
        return self.vindex[self.pcol(p_idx, self.vcols[u])]

    def orbit_cols(self, f):
        out = []
        for pi in range(len(PERMS)):
            g = self.pcol(pi, f)
            if g not in out:
                out.append(g)
        return out

    def colmat(self):
        if self._C is None or self._C.shape != (len(self.row_col), self.n):
            self._C = np.array(self.row_col, dtype=np.int64).reshape(len(self.row_col), self.n)
        return self._C

    def _new_edges(self, new_vertices):
        P = self.P
        new_vertices = set(new_vertices)
        V = np.array(self.vcols)
        C = self.colmat() if self.row_col else None
        for v in sorted(new_vertices):
            for u in range(self.n):
                if u == v or (u in new_vertices and u > v):
                    continue
                a, b = min(u, v), max(u, v)
                if (a, b) in self.edge_set:
                    continue
                orb = []
                for pi in range(len(PERMS)):
                    pa, pb = self.vperm(pi, a), self.vperm(pi, b)
                    e = (min(pa, pb), max(pa, pb))
                    if e not in orb:
                        orb.append(e)
                oid = len(self.ovar)
                for e in orb:
                    self.edge_set[e] = oid
                    self.eu.append(e[0]); self.ev.append(e[1]); self.eorb.append(oid)
                ou = np.array([e[0] for e in orb]); ov = np.array([e[1] for e in orb])
                comp = (V[ou] != V[ov]).sum(0).astype(float)
                rows = [self.comp_rows[s] for s in np.nonzero(comp)[0]]
                vals = list(comp[comp != 0])
                if C is not None:
                    cnt = (C[:, ou] != C[:, ov]).sum(1).astype(float)
                    nz = np.nonzero(cnt)[0]
                    rows += [self.rows[r] for r in nz]
                    vals += list(-cnt[nz])
                self.ovar.append(P.add_col(rows, vals))

    def add_orbit(self, f, fills=None):
        members = [g for g in self.orbit_cols(f) if g not in self.vindex]
        if not members:
            return []
        new = []
        for g in members:
            v = self.n
            self.vcols.append(g)
            self.vindex[g] = v
            self.alive.append(True)
            fill = None if fills is None else fills.get(g)
            if fill is None:
                fill = self.P.rng.integers(0, self.k, size=len(self.row_col))
            for r, c in enumerate(fill):
                self.row_col[r].append(int(c))
            new.append(v)
        self._C = None
        self._new_edges(new)
        return new

    def add_sound_row(self, col, sat, y):
        key = tuple(int(c) for c in col)
        if key in self.row_key:
            return False
        self.row_key.add(key)
        col = np.array(key)
        eu, ev, eo = np.array(self.eu), np.array(self.ev), np.array(self.eorb)
        cnt = np.bincount(eo[col[eu] != col[ev]], minlength=len(self.ovar)).astype(float)
        nz = np.nonzero(cnt)[0]
        idx = [self.ovar[o] for o in nz] + [self.P.s_idx(self.i, tuple(y), 1 if sat else 2)]
        vals = list(-cnt[nz]) + [1.0]
        self.rows.append(self.P.add_row(idx, vals, 0.0, INF))
        self.row_col.append(list(key))
        self.row_sat.append(bool(sat))
        self._C = None
        return True

    def orbit_w(self, sol):
        return np.maximum(sol[np.array(self.ovar)], 0.0)

    def W(self, sol=None):
        sol = self.P.sol if sol is None else sol
        w = self.orbit_w(sol)
        W = np.zeros((self.n, self.n))
        eu, ev, eo = np.array(self.eu), np.array(self.ev), np.array(self.eorb)
        W[eu, ev] = w[eo]
        W[ev, eu] = w[eo]
        return W

    def used_vertices(self, tol=1e-9):
        W = self.W()
        return [u for u in range(self.nf, self.n) if W[u].sum() > tol]

    def separate(self, restarts=8, rounds=3, tol=1e-7):
        sol = self.P.sol
        W = self.W()
        vals, cols = batched_local_search(W, self.k, self.fixed, self.nf, restarts=restarts,
                                          rounds=rounds, rng=self.P.rng)
        viol = vals - sol[self.svar]
        added = 0
        for j in np.argsort(-viol):
            if viol[j] <= tol:
                break
            if self.add_sound_row(cols[j], self.is_sat[j], self.items[j][1]):
                added += 1
        return added, float(max(viol.max(), 0.0))

    def seed_random(self):
        for j in range(len(self.items)):
            col = np.concatenate([self.fixed[j], self.P.rng.integers(0, self.k, size=self.n - self.nf)])
            self.add_sound_row(col, self.is_sat[j], self.items[j][1])

    # pricing -------------------------------------------------------------
    def duals(self):
        d = self.P.dual
        return np.abs(d[np.array(self.comp_rows)]), np.abs(d[np.array(self.rows)])

    def price(self, N, d, rng=None, act_rows=None):
        rng = self.P.rng if rng is None else rng
        k = self.k
        pi_d, y = self.duals()
        V = np.array(self.vcols)
        d = np.asarray(d, dtype=float)
        D = d.sum()
        same = np.stack([(d[:, None] * (V[N] == c)).sum(0) for c in range(k)], 1)
        best = same.min(1)
        f = tuple(int(rng.choice(np.nonzero(same[s] <= best[s] + 1e-12)[0])) for s in range(self.S))
        C = self.colmat()
        rows = np.nonzero(y > 1e-12)[0] if act_rows is None else act_rows
        total, fills, seen = 0.0, {}, set()
        for pi in range(len(PERMS)):
            g = self.pcol(pi, f)
            if g in seen:
                continue
            seen.add(g)
            Np = [self.vperm(pi, u) for u in N]
            gv = np.array(g)
            comp = (D - (d[:, None] * (V[Np] == gv[None, :])).sum(0)) @ pi_d
            Cr = C[rows][:, Np]
            sameR = np.stack([((Cr == c) * d[None, :]).sum(1) for c in range(k)], 1)
            total += comp - ((D - sameR.min(1)) * y[rows]).sum()
            CA = C[:, Np]
            sameA = np.stack([((CA == c) * d[None, :]).sum(1) for c in range(k)], 1)
            fills[g] = sameA.argmin(1)
        return total / (D * len(seen)), f, fills


class JointSymLP:
    def __init__(self, k, columns_per_i, refined=True, seed=0):
        self.k = k
        self.refined = refined
        self.rng = np.random.default_rng(seed)
        self.Y = allowed_global_assignments(k)
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        h.setOptionValue('presolve', 'off')
        self.h = h
        self.nvar = 0
        self.nrow = 0
        self.tau = self._var(0.0, INF, 1.0)
        self._c = [self._var(0.0, INF) for _ in range(k)]
        self._s = {}
        for i in range(k):
            for t in (1, 2):
                if refined:
                    for y in self.Y:
                        self._s[(i, y, t)] = self._var(-INF, INF)
                else:
                    v = self._var(-INF, INF)
                    for y in self.Y:
                        self._s[(i, y, t)] = v
        self.add_row([self.c_idx(i) for i in range(k)], [1.0] * k, 1.0, INF)
        Ys = self.Y if refined else self.Y[:1]
        for y in Ys:
            base = [self.s_idx(i, y, 2) for i in range(k)]
            self.add_row([self.tau] + base, [1.0] + [-1.0] * k, 0.0, INF)
            for r in range(k):
                idx = [self.tau] + base + [self.s_idx(r, y, 1)]
                val = [1.0] + [-1.0] * k + [-1.0]
                val[1 + r] += 1.0           # + s2[r,y]
                self.add_row(idx, val, 0.0, INF)
        self.blocks = [Block(self, i) for i in range(k)]
        for i in range(k):
            for f in columns_per_i[i]:
                self.blocks[i].add_orbit(tuple(int(t) for t in f))
        self.sol = None

    def _var(self, lo, hi, cost=0.0):
        self.h.addVar(lo, hi)
        if cost:
            self.h.changeColCost(self.nvar, cost)
        self.nvar += 1
        return self.nvar - 1

    def c_idx(self, i):
        return self._c[i]

    def s_idx(self, i, y, t):
        return self._s[(i, tuple(y), t)]

    def add_row(self, idx, vals, lo, hi):
        # merge duplicate indices
        acc = {}
        for j, v in zip(idx, vals):
            acc[j] = acc.get(j, 0.0) + v
        idx = [j for j in acc if acc[j] != 0.0]
        vals = [acc[j] for j in idx]
        self.h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.array(vals, dtype=np.float64))
        self.nrow += 1
        return self.nrow - 1

    def add_col(self, rows, vals):
        self.h.addCol(0.0, 0.0, INF, len(rows), np.array(rows, dtype=np.int32), np.array(vals, dtype=np.float64))
        self.nvar += 1
        return self.nvar - 1

    def solve(self):
        self.h.run()
        st = self.h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(str(st))
        s = self.h.getSolution()
        self.sol = np.array(s.col_value)
        self.dual = np.array(s.row_dual)
        return self.sol[self.tau]

    def optimize(self, max_iter=300, restarts=8, rounds=3, verbose=False):
        for b in self.blocks:
            if not b.row_col:
                b.seed_random()
        for it in range(max_iter):
            tau = self.solve()
            added = 0
            worst = 0.0
            for b in self.blocks:
                a, w = b.separate(restarts=restarts, rounds=rounds)
                added += a
                worst = max(worst, w)
            if verbose:
                print(f'    it {it}: tau={tau:.7f} added={added} worst={worst:.2g} rows={self.nrow}', flush=True)
            if added == 0:
                break
        return self.solve()


def joint_colgen_round(J, top=3, n_samples=1200):
    """One round of dual-priced column generation in every block of the joint LP."""
    from .symlp import generate_sym
    added = 0
    for b in J.blocks:
        for rc, f, fills, N, d in generate_sym(b, n_samples=n_samples, top=top):
            b.add_orbit(f, fills=fills)
            added += 1
    return added
