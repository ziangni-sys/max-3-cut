"""S_3-symmetric version of the dynamic single-predicate gadget LP.

The i-gadget problem (sat set C_i, global patterns Y) is invariant under permuting the
three primary variables, which acts on columns by (pi f)(x) = f(pi^{-1} x).  For a
column set closed under this action the LP optimum is attained by a symmetric gadget
(average over the group), so we use one LP variable per *edge orbit*, add auxiliary
vertices as whole orbits, and separate only over x up to permutation.
"""
from itertools import permutations, product
import numpy as np
import highspy

from .core import allowed_global_assignments, sat_assignments
from .search import batched_local_search

INF = highspy.kHighsInf
PERMS = list(permutations(range(3)))


def act_x(p, x):
    """(p . x)_{p[j]} = x_j"""
    y = [0, 0, 0]
    for j in range(3):
        y[p[j]] = x[j]
    return tuple(y)


def inv(p):
    q = [0, 0, 0]
    for j in range(3):
        q[p[j]] = j
    return tuple(q)


class SymDynLP:
    def __init__(self, k, i, columns=(), seed=0, excess=0.0, Y=None):
        self.k, self.i = k, i
        self.rng = np.random.default_rng(seed)
        self.sat = sat_assignments(k, i)
        self.S = len(self.sat)
        self.sidx = {x: j for j, x in enumerate(self.sat)}
        # sat-index permutation for f -> p.f :  (p.f)[j] = f[sidx[p^{-1} sat[j]]]
        self.colperm = [np.array([self.sidx[act_x(inv(p), x)] for x in self.sat]) for p in PERMS]
        self.Y = allowed_global_assignments(k) if Y is None else [tuple(y) for y in Y]
        reps = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3)))
        self.items = [(x, y) for x in reps for y in self.Y]
        self.fixed = np.array([list(x) + list(y) for x, y in self.items], dtype=np.int64)
        self.is_sat = np.array([sum(x) % k == i for x, _ in self.items])
        self.excess = excess
        prim = [tuple(x[j] for x in self.sat) for j in range(3)]
        const = [tuple([c] * self.S) for c in range(k)]
        self.vcols = prim + const
        self.vindex = {f: u for u, f in enumerate(self.vcols)}
        self.alive = [True] * len(self.vcols)
        self.nf = 3 + k
        self.h = highspy.Highs()
        self.h.setOptionValue('output_flag', False)
        self.h.setOptionValue('presolve', 'off')
        self.h.addVar(-INF, INF)
        self.h.changeColCost(0, -1.0)
        for _ in range(self.S):
            self.h.addRow(1.0, INF, 0, np.array([], dtype=np.int32), np.array([]))
        self.eu, self.ev, self.eorb = [], [], []    # edge list and orbit id
        self.edge_set = {}
        self.norb = 0
        self.row_col, self.row_sat, self.row_key = [], [], set()
        self._C = None
        self.sol = None
        self._new_edges(range(self.n))
        for f in columns:
            self.add_orbit(tuple(int(t) for t in f))

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

    def _colmat(self):
        if self._C is None or self._C.shape != (len(self.row_col), self.n):
            self._C = np.array(self.row_col, dtype=np.int64).reshape(len(self.row_col), self.n)
        return self._C

    def _new_edges(self, new_vertices):
        """Create orbit variables for all edges touching new_vertices."""
        new_vertices = set(new_vertices)
        V = np.array(self.vcols)
        C = self._colmat() if self.row_col else None
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
                oid = self.norb
                self.norb += 1
                for e in orb:
                    self.edge_set[e] = oid
                    self.eu.append(e[0])
                    self.ev.append(e[1])
                    self.eorb.append(oid)
                ou = np.array([e[0] for e in orb])
                ov = np.array([e[1] for e in orb])
                comp = (V[ou] != V[ov]).sum(0).astype(float)          # per sat x
                rows = list(np.nonzero(comp)[0])
                vals = list(comp[comp != 0])
                if C is not None:
                    cnt = (C[:, ou] != C[:, ov]).sum(1).astype(float)
                    nz = np.nonzero(cnt)[0]
                    rows += list(self.S + nz)
                    vals += list(cnt[nz])
                self.h.addCol(0.0, 0.0, INF, len(rows), np.array(rows, dtype=np.int32), np.array(vals))
                if not all(self.alive[w] for e in orb for w in e):
                    self.h.changeColBounds(1 + oid, 0.0, 0.0)

    def add_orbit(self, f, fills=None):
        """Add the S_3-orbit of column f.  fills: dict column -> fill colors for existing rows."""
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
                fill = self.rng.integers(0, self.k, size=len(self.row_col))
            for r, c in enumerate(fill):
                self.row_col[r].append(int(c))
            new.append(v)
        self._C = None
        self._new_edges(new)
        return new

    def add_sound_row(self, col, sat):
        key = tuple(int(c) for c in col)
        if key in self.row_key:
            return False
        self.row_key.add(key)
        col = np.array(key)
        eu, ev, eo = np.array(self.eu), np.array(self.ev), np.array(self.eorb)
        cnt = np.bincount(eo[col[eu] != col[ev]], minlength=self.norb).astype(float)
        idx = list(1 + np.nonzero(cnt)[0])
        vals = list(cnt[cnt != 0])
        if not sat:
            idx.append(0)
            vals.append(1.0)
        self.h.addRow(-INF, 1.0 + (self.excess if sat else 0.0), len(idx),
                      np.array(idx, dtype=np.int32), np.array(vals))
        self.row_col.append(list(key))
        self.row_sat.append(bool(sat))
        self._C = None
        return True

    def solve(self):
        self.h.run()
        st = self.h.getModelStatus()
        if st != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(str(st))
        s = self.h.getSolution()
        self.sol = np.array(s.col_value)
        self.dual = np.array(s.row_dual)
        return self.sol[0]

    def orbit_w(self):
        return np.maximum(self.sol[1:1 + self.norb], 0.0)

    def W(self):
        w = self.orbit_w()
        W = np.zeros((self.n, self.n))
        eu, ev, eo = np.array(self.eu), np.array(self.ev), np.array(self.eorb)
        W[eu, ev] = w[eo]
        W[ev, eu] = w[eo]
        return W

    def bounds(self):
        g = self.sol[0]
        return np.where(self.is_sat, 1.0 + self.excess, 1.0 - g)

    def add_sound_rows(self, cols, sats):
        """Batch version of add_sound_row (vectorized coefficients, one HiGHS call)."""
        keep_c, keep_s = [], []
        for col, sat in zip(cols, sats):
            key = tuple(int(c) for c in col)
            if key in self.row_key:
                continue
            self.row_key.add(key)
            keep_c.append(key)
            keep_s.append(bool(sat))
        if not keep_c:
            return 0
        C = np.array(keep_c, dtype=np.int64)
        eu, ev, eo = np.array(self.eu), np.array(self.ev), np.array(self.eorb)
        mask = (C[:, eu] != C[:, ev]).astype(np.float64)            # m x E
        cnt = np.zeros((len(keep_c), self.norb))
        np.add.at(cnt.T, eo, mask.T)                                  # m x norb
        starts, idx, vals, lo, hi = [], [], [], [], []
        for r in range(len(keep_c)):
            nz = np.nonzero(cnt[r])[0]
            starts.append(len(idx))
            idx.extend((1 + nz).tolist())
            vals.extend(cnt[r, nz].tolist())
            if not keep_s[r]:
                idx.append(0)
                vals.append(1.0)
            lo.append(-INF)
            hi.append(1.0 + (self.excess if keep_s[r] else 0.0))
        self.h.addRows(len(keep_c), np.array(lo), np.array(hi), len(idx),
                       np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32), np.array(vals))
        for key, sat in zip(keep_c, keep_s):
            self.row_col.append(list(key))
            self.row_sat.append(sat)
        self._C = None
        return len(keep_c)

    def separate(self, restarts=8, rounds=3, tol=1e-7, max_add=300):
        W = self.W()
        vals, cols = batched_local_search(W, self.k, self.fixed, self.nf,
                                          restarts=restarts, rounds=rounds, rng=self.rng)
        viol = vals - self.bounds()
        order = [j for j in np.argsort(-viol) if viol[j] > tol][:max_add]
        added = self.add_sound_rows([cols[j] for j in order], [self.is_sat[j] for j in order])
        return added, float(max(viol.max(), 0.0))

    def seed_random(self):
        cols = [np.concatenate([self.fixed[j], self.rng.integers(0, self.k, size=self.n - self.nf)])
                for j in range(len(self.items))]
        self.add_sound_rows(cols, list(self.is_sat))

    def prune_rows(self, max_rows=6000, slack_tol=1e-4):
        """Delete soundness rows with slack (they can be re-added by separation)."""
        if len(self.row_col) <= max_rows:
            return 0
        act = np.array(self.h.getSolution().row_value)[self.S:]
        ub = np.where(np.array(self.row_sat), 1.0 + self.excess, 1.0)
        slack = ub - act
        order = np.argsort(slack)                       # keep the tightest max_rows/2
        keep = set(order[: max_rows // 2].tolist()) | set(np.nonzero(slack < slack_tol)[0].tolist())
        drop = np.array(sorted(set(range(len(self.row_col))) - keep), dtype=np.int32)
        if len(drop) == 0:
            return 0
        self.h.deleteRows(len(drop), self.S + drop)
        dropset = set(drop.tolist())
        for r in drop.tolist():
            self.row_key.discard(tuple(self.row_col[r]))
        self.row_col = [c for r, c in enumerate(self.row_col) if r not in dropset]
        self.row_sat = [t for r, t in enumerate(self.row_sat) if r not in dropset]
        self._C = None
        return len(drop)

    def optimize(self, max_iter=300, restarts=8, rounds=3, verbose=False, rel_tol=1e-7, max_rows=6000):
        if not self.row_col:
            self.seed_random()
        for it in range(max_iter):
            g = self.solve()
            if self.prune_rows(max_rows=max_rows):
                g = self.solve()
            added, worst = self.separate(restarts=restarts, rounds=rounds, tol=max(1e-7, rel_tol * abs(g)))
            if verbose:
                print(f'    it {it}: g={g:.7f} added={added} worst={worst:.2g} rows={len(self.row_col)}', flush=True)
            if added == 0:
                break
        return self.solve()

    def duals(self):
        d = self.dual
        return np.abs(d[:self.S]), np.abs(d[self.S:])

    def used_vertices(self, tol=1e-9):
        W = self.W()
        return [u for u in range(self.nf, self.n) if self.alive[u] and W[u].sum() > tol]

    def kill_orbit(self, u):
        """Fix to zero every edge-orbit variable touching the vertex orbit of u."""
        members = {self.vindex[g] for g in self.orbit_cols(self.vcols[u])}
        for m in members:
            self.alive[m] = False
        dead = set()
        for a, b, o in zip(self.eu, self.ev, self.eorb):
            if a in members or b in members:
                dead.add(o)
        for o in dead:
            self.h.changeColBounds(1 + o, 0.0, 0.0)
        return members

    def prune_to(self, cap):
        """Kill lowest-weight vertex orbits until at most cap auxiliary vertices carry weight."""
        killed = 0
        while True:
            used = self.used_vertices()
            if len(used) <= cap:
                return killed
            W = self.W()
            orbs = {}
            for u in used:
                key = min(self.orbit_cols(self.vcols[u]))
                orbs.setdefault(key, []).append(u)
            worst = min(orbs.values(), key=lambda us: W[us].sum())
            self.kill_orbit(worst[0])
            killed += 1
            self.optimize(rel_tol=1e-3)

    # ---------------------------------------------------------------- pricing
    def price(self, N, d, rng=None, act_rows=None):
        """RC of adding the orbit of a vertex joined to N (weights d); its column is a
        best response (random tie-breaking).  Returns (rc per unit weight, f, fills)."""
        rng = self.rng if rng is None else rng
        k = self.k
        pi_d, y = self.duals()
        V = np.array(self.vcols)
        d = np.asarray(d, dtype=float)
        D = d.sum()
        same = np.stack([(d[:, None] * (V[N] == c)).sum(0) for c in range(k)], 1)
        best = same.min(1)
        f = []
        for s in range(self.S):
            cands = np.nonzero(same[s] <= best[s] + 1e-12)[0]
            f.append(int(rng.choice(cands)))
        f = tuple(f)
        C = self._colmat()
        rows = np.nonzero(y > 1e-12)[0] if act_rows is None else act_rows
        total = 0.0
        fills = {}
        seen = set()
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
            snd = ((D - sameR.min(1)) * y[rows]).sum()
            total += comp - snd
            CA = C[:, Np]
            sameA = np.stack([((CA == c) * d[None, :]).sum(1) for c in range(k)], 1)
            fills[g] = sameA.argmin(1)
        return total / (D * len(seen)), f, fills


def generate_sym(lp, n_samples=1500, max_support=5, top=3, hill=25, wmax=3):
    rng = lp.rng
    alive = [u for u in range(lp.n) if lp.alive[u]]
    W = lp.W()
    deg = W.sum(1)[alive]
    _, y = lp.duals()
    act = np.nonzero(y > 1e-12)[0]
    p = deg + 0.05 * deg.max() + 1e-9
    p = p / p.sum()
    cand = []
    for _ in range(n_samples):
        s = int(rng.integers(2, max_support + 1))
        N = list(rng.choice(alive, size=min(s, len(alive)), replace=False, p=p))
        d = list(rng.integers(1, wmax + 1, size=len(N)))
        rc, f, _ = lp.price(N, d, rng=rng, act_rows=act)
        cand.append((rc, N, d))
    cand.sort(key=lambda t: -t[0])
    out, seen = [], set(lp.vcols)
    for rc0, N, d in cand[:top * 3]:
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
            if rc > best[0]:
                best = (rc, N2, d2)
        rc, N, d = best
        if rc <= 1e-9:
            continue
        for _ in range(3):
            _, f, fills = lp.price(N, d, rng=rng)
            if f not in seen:
                for g in lp.orbit_cols(f):
                    seen.add(g)
                out.append((rc, f, fills, N, d))
                break
        if len(out) >= top:
            break
    return out
