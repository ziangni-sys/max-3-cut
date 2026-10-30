"""Certification of a gadget family for NRT's Theorem A.4 (original theorem), k >= 2.

Needed: C_i (lower bound, explicit witnesses), S1_i, S2_i (upper bounds) where S1/S2 are maxima
over ALL y in Y.  Instances (x, y) are taken up to the checked S_3 automorphisms and the
permutations of colours unused by y; an orbit can contain satisfying and unsatisfying x.
Upper bounds per instance: (1) rigorous SDP bound (gadgets/sdpbound.py: valid for any dual
multipliers, eigenvalue-corrected); if it is <= the class threshold the instance is closed;
(2) otherwise CP-SAT (strong model + value precedence) proves  value <= threshold  or finds a
better coloring, which raises the threshold (then re-proved).  SDP and CP-SAT run in separate
spawned processes (cvxpy loads HiGHS, which clashes with OR-tools).  Resumable cache.
Usage: verify_a4.py family.json [procs]
"""
import sys, os, json, time, hashlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from fractions import Fraction
from itertools import product, permutations
import multiprocessing as mp
import numpy as np


def require(cond, msg):
    if not cond:
        raise ValueError(msg)


def sdp_job(a):
    from gadgets.sdpbound import sdp_upper_bound
    gi, x, y, W, k, fixed = a
    t0 = time.time()
    ub, pv, st = sdp_upper_bound(W, k, list(fixed), 3 + k)
    return gi, x, y, ub, time.time() - t0


def cp_job(a):
    from gadgets.cpsat import cpsat_decide_strong
    gi, x, y, W, k, fixed, thr, hint = a
    t0 = time.time()
    while True:
        r = cpsat_decide_strong(W, k, list(fixed), 3 + k, thr, hint=hint, time_limit=6 * 3600.0,
                                value_precedence=True)
        if r is None:
            return gi, x, y, thr, time.time() - t0
        thr, hint = r[0], r[1]


def main():
    from gadgets.core import edges_to_matrix, allowed_global_assignments
    from gadgets.localsearch import batched_local_search
    path = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ctx = mp.get_context('spawn')
    d = json.load(open(path))
    k = d['k']
    require(sorted(g['i'] for g in d['gadgets']) == list(range(k)), 'one gadget per predicate')
    Y = allowed_global_assignments(k)
    y0 = tuple(range(k))
    allx = list(product(range(k), repeat=3))
    cache_path = path.replace('.json', '.a4cache.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            e = json.loads(line)
            cache[(e['hash'], tuple(e['x']), tuple(e['y']))] = e
    out = []
    for g in d['gadgets']:
        t0 = time.time()
        i = g['i']
        require(all(isinstance(w, int) and w > 0 and a != b for a, b, w in g['edges']), 'bad edges')
        W = edges_to_matrix(g['edges'], n=g['n_vertices'])
        hsh = hashlib.sha256(np.ascontiguousarray(W, dtype=np.int64).tobytes()).hexdigest()[:16]
        n = W.shape[0]
        sym = False
        if g.get('automorphisms'):
            realized = set()
            for P in g['automorphisms']:
                P = [p - 1 for p in P]
                require(sorted(P) == list(range(n)), 'not a permutation')
                require(all(P[3 + j] == 3 + j for j in range(k)), 'must fix globals')
                require(sorted(P[:3]) == [0, 1, 2], 'must permute primaries')
                Pa = np.array(P)
                require(np.array_equal(W[np.ix_(Pa, Pa)], W), 'not an automorphism')
                realized.add(tuple(P[:3]))
            sym = realized == set(permutations(range(3)))
        keys, classes = [], []
        for y in Y:
            free = [c for c in range(k) if c not in set(y)]
            maps = []
            for pf in permutations(free):
                mm = list(range(k))
                for a_, b_ in zip(free, pf):
                    mm[a_] = b_
                maps.append(mm)
            seen = set()
            for x in allx:
                if x in seen:
                    continue
                orb = set()
                for mm in maps:
                    xm = tuple(mm[v] for v in x)
                    orb |= set(permutations(xm)) if sym else {xm}
                seen |= orb
                keys.append((min(orb), y))
                classes.append(sorted({sum(v) % k == i for v in orb}))
        fixed = np.array([list(x) + list(y) for x, y in keys])
        lv, lc = batched_local_search(W.astype(float), k, fixed, 3 + k, restarts=24, rounds=5,
                                      rng=np.random.default_rng(12345))
        lv = np.rint(lv).astype(np.int64)
        # completeness witnesses (exact integer re-evaluation)
        satx = [x for x in allx if sum(x) % k == i]
        fx = np.array([list(x) + list(y0) for x in satx])
        cv, ccol = batched_local_search(W.astype(float), k, fx, 3 + k, restarts=24, rounds=5,
                                        rng=np.random.default_rng(54321))
        wit = []
        for x, col in zip(satx, ccol):
            col = np.array(col)
            require(list(col[:3 + k]) == list(x) + list(y0), 'witness must keep primaries/globals')
            wit.append(int((W * (col[:, None] != col[None, :])).sum() // 2))
        C = min(wit)
        T = {True: -1, False: -1}
        for cl, v in zip(classes, lv):
            for c in cl:
                T[c] = max(T[c], int(v))
        for (x, y), cl in zip(keys, classes):          # proven values from the cache raise T
            e = cache.get((hsh, x, y))
            if e is not None and e['how'] == 'cpsat':
                for c in cl:
                    T[c] = max(T[c], e['bound'])
        cf = open(cache_path, 'a')
        # phase 1: SDP
        todo = [(i, x, y, W, k, f) for (x, y), f in zip(keys, fixed) if (hsh, x, y) not in cache]
        nsdp = 0
        with ctx.Pool(procs) as p:
            for gi, x, y, ub, dt in p.imap_unordered(sdp_job, todo, chunksize=2):
                e = {'hash': hsh, 'i': gi, 'x': list(x), 'y': list(y), 'how': 'sdp', 'bound': float(ub), 'secs': round(dt, 2)}
                cache[(hsh, x, y)] = e
                cf.write(json.dumps(e) + '\n'); cf.flush()
                nsdp += 1
        # phase 2: CP-SAT for instances whose SDP bound exceeds the threshold
        def thr_of(cl):
            return min(T[c] for c in cl)
        todo = []
        for (x, y), cl, f, h in zip(keys, classes, fixed, lc):
            e = cache[(hsh, x, y)]
            if e['how'] == 'sdp' and e['bound'] <= thr_of(cl):
                continue
            if e['how'] == 'cpsat' and e['bound'] <= thr_of(cl):
                continue
            todo.append((i, x, y, W, k, f, thr_of(cl), h))
        print(f'i={i}: {len(keys)} orbit instances; SDP closed {len(keys) - len(todo)}; CP-SAT needed for {len(todo)}', flush=True)
        with ctx.Pool(procs) as p:
            for gi, x, y, proved, dt in p.imap_unordered(cp_job, todo, chunksize=1):
                e = {'hash': hsh, 'i': gi, 'x': list(x), 'y': list(y), 'how': 'cpsat', 'bound': int(proved), 'secs': round(dt, 1)}
                cache[(hsh, x, y)] = e
                cf.write(json.dumps(e) + '\n'); cf.flush()
        cf.close()
        # final: every instance has an upper bound; S1/S2 = max over orbits of their bounds,
        # but an orbit closed by SDP contributes only its threshold-certified status: take
        # S_c = max(T_c, max of CP-SAT-proved values in class c)
        S = dict(T)
        for (x, y), cl in zip(keys, classes):
            e = cache[(hsh, x, y)]
            if e['how'] == 'cpsat':
                for c in cl:
                    S[c] = max(S[c], e['bound'])
        for (x, y), cl in zip(keys, classes):           # sanity: every instance is covered
            e = cache[(hsh, x, y)]
            require(e['bound'] <= min(S[c] for c in cl) + 1e-6, f'uncovered instance {x} {y}')
        t = int(np.triu(W, 1).sum())
        out.append(dict(i=i, C=C, S1=S[True], S2=S[False], t=t, hash=hsh,
                        witnesses=[{'x': list(x), 'coloring': [int(c) for c in col], 'value': v}
                                   for x, col, v in zip(satx, ccol, wit)]))
        print(f'i={i}: C>={C} S1<={S[True]} S2<={S[False]} t={t} n_aux={n-3-k} sym={sym} ({time.time()-t0:.0f}s)', flush=True)
    sumC = sum(p['C'] for p in out)
    ratio = Fraction(sum(p['S2'] for p in out) + max(0, max(p['S1'] - p['S2'] for p in out)), sumC)
    print(f'certified NRT Thm A.4 ratio <= {ratio} = {float(ratio):.7f}', flush=True)
    json.dump({'nrt_ratio': [ratio.numerator, ratio.denominator],
               'gadgets': [{'i': p['i'], 'hash': p['hash'], 'C_lower': p['C'], 'S1_upper': p['S1'],
                            'S2_upper': p['S2'], 't': p['t'], 'completeness_witnesses': p['witnesses']} for p in out]},
              open(path.replace('.json', '.a4cert.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
