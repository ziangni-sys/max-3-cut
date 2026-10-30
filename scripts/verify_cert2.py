"""Resumable certificate verification of a gadget family (strong CP-SAT model).

Same logic as verify_cert.py: checked automorphisms -> x up to permutation; lower bounds
from explicit colorings; every upper bound proved by CP-SAT (INFEASIBLE for "value >
threshold").  Each proved instance is appended to <family>.cache.jsonl immediately, so
a restart resumes where it stopped.
Usage: verify_cert2.py family.json [procs]
"""
import sys, os, json, time, hashlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from fractions import Fraction
from itertools import product, permutations
from multiprocessing import Pool
import numpy as np
from gadgets.core import edges_to_matrix, allowed_global_assignments
from gadgets.cpsat import cpsat_decide_strong
from gadgets.localsearch import batched_local_search


def job(a):
    gi, x, y, W, k, fixed, thr, hint = a
    t0 = time.time()
    while True:
        r = cpsat_decide_strong(W, k, list(fixed), 3 + k, thr, hint=hint, time_limit=6 * 3600.0)
        if r is None:
            return gi, x, y, thr, time.time() - t0
        thr, hint = r[0], r[1]          # found a better coloring: raise and re-prove


def require(cond, msg):
    if not cond:
        raise ValueError(msg)


def check_automorphisms(W, k, perms):
    n = W.shape[0]
    realized = set()
    for P in perms:
        P = [p - 1 for p in P]
        require(sorted(P) == list(range(n)), 'not a permutation')
        require(all(P[3 + j] == 3 + j for j in range(k)), 'automorphism must fix the globals')
        require(sorted(P[:3]) == [0, 1, 2], 'automorphism must permute the primaries')
        Pa = np.array(P)
        require(np.array_equal(W[np.ix_(Pa, Pa)], W), 'not an automorphism')
        realized.add(tuple(P[:3]))
    return realized


def gadget_hash(W):
    return hashlib.sha256(np.ascontiguousarray(W, dtype=np.int64).tobytes()).hexdigest()[:16]


def main():
    path = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cache_path = path.replace('.json', '.cache.jsonl')
    d = json.load(open(path))
    k = d['k']
    require(sorted(g['i'] for g in d['gadgets']) == list(range(k)), 'need exactly one gadget per predicate')
    for g in d['gadgets']:
        require(all(isinstance(w, int) and w > 0 and a != b for a, b, w in g['edges']),
                'weights must be positive integers, no self-loops')
    hashes = {g['i']: gadget_hash(edges_to_matrix(g['edges'], n=g['n_vertices'])) for g in d['gadgets']}
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            e = json.loads(line)
            if e.get('hash') == hashes.get(e['i']):          # ignore entries of other gadgets
                cache[(e['i'], tuple(e['x']), tuple(e['y']))] = e['proved_max']
    Y = allowed_global_assignments(k)
    y0 = tuple(range(k))
    out = []
    for g in d['gadgets']:
        t0 = time.time()
        i = g['i']
        W = edges_to_matrix(g['edges'], n=g['n_vertices'])
        sym = bool(g.get('automorphisms')) and \
            check_automorphisms(W, k, g['automorphisms']) == set(permutations(range(3)))
        X = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3))) if sym \
            else list(product(range(k), repeat=3))
        keys = [(x, y) for x in X for y in Y]
        fixed = np.array([list(x) + list(y) for x, y in keys])
        lv, lc = batched_local_search(W.astype(float), k, fixed, 3 + k, restarts=24, rounds=5,
                                      rng=np.random.default_rng(12345))
        lv = np.rint(lv).astype(np.int64)
        satx = [x for x in product(range(k), repeat=3) if sum(x) % k == i]
        fx = np.array([list(x) + list(y0) for x in satx])
        cv, ccol = batched_local_search(W.astype(float), k, fx, 3 + k, restarts=24, rounds=5,
                                        rng=np.random.default_rng(54321))
        # completeness witnesses, re-evaluated exactly in integers
        wit = []
        for x, col in zip(satx, ccol):
            col = np.array(col)
            require(list(col[:3 + k]) == list(x) + list(y0), 'witness must keep primaries/globals')
            wit.append(int((W * (col[:, None] != col[None, :])).sum() // 2))
        C = min(wit)
        witnesses = [{'x': list(x), 'coloring': [int(c) for c in col], 'value': v}
                     for x, col, v in zip(satx, ccol, wit)]
        thr = {}
        for (x, y), v in zip(keys, lv):
            key = (sum(x) % k == i, y)
            thr[key] = max(thr.get(key, -1), int(v))
        # cached proofs: proved_max >= current threshold is reusable (raise thr if bigger)
        for (x, y) in keys:
            c = cache.get((i, x, y))
            if c is not None:
                key = (sum(x) % k == i, y)
                thr[key] = max(thr[key], c)
        todo = [(i, x, y, W, k, f, thr[(sum(x) % k == i, y)], h)
                for (x, y), f, h in zip(keys, fixed, lc) if (i, x, y) not in cache]
        print(f'i={i}: {len(keys)} instances, {len(keys) - len(todo)} cached, sym={sym}', flush=True)
        with Pool(procs) as p, open(cache_path, 'a') as cf:
            for gi, x, y, proved, dt in p.imap_unordered(job, todo, chunksize=1):
                cache[(gi, x, y)] = proved
                cf.write(json.dumps({'i': gi, 'hash': hashes[gi], 'x': list(x), 'y': list(y), 'proved_max': int(proved),
                                     'secs': round(dt, 1)}) + '\n')
                cf.flush()
        # final thresholds: every instance proved <= its recorded bound; take max per class
        S1y = {y: max(cache[(i, x, y)] for x in X if sum(x) % k == i) for y in Y}
        S2y = {y: max(cache[(i, x, y)] for x in X if sum(x) % k != i) for y in Y}
        t = int(np.triu(W, 1).sum())
        out.append(dict(i=i, C=C, S1y=S1y, S2y=S2y, S1=max(S1y.values()), S2=max(S2y.values()), t=t,
                        witnesses=witnesses, hash=hashes[i]))
        print(f'i={i}: C>={C} S1<={max(S1y.values())} S2<={max(S2y.values())} t={t} '
              f'n_aux={W.shape[0]-3-k} ({time.time()-t0:.0f}s)', flush=True)
    sumC = sum(p['C'] for p in out)
    nrt = Fraction(sum(p['S2'] for p in out) + max(0, max(p['S1'] - p['S2'] for p in out)), sumC)
    ref = Fraction(max(sum(p['S2y'][y] for p in out) + max(0, max(p['S1y'][y] - p['S2y'][y] for p in out))
                       for y in Y), sumC)
    print(f'certified NRT Thm A.4 ratio      <= {nrt} = {float(nrt):.7f}')
    print(f'certified shared-y refined ratio <= {ref} = {float(ref):.7f}')
    json.dump({'nrt_ratio': [nrt.numerator, nrt.denominator], 'refined_ratio': [ref.numerator, ref.denominator],
               'gadgets': [{'i': p['i'], 'hash': p['hash'], 'C_lower': p['C'], 'S1_upper': p['S1'], 'S2_upper': p['S2'], 't': p['t'],
                            'completeness_witnesses': p['witnesses'],
                            'S1y': {''.join(map(str, y)): v for y, v in p['S1y'].items()},
                            'S2y': {''.join(map(str, y)): v for y, v in p['S2y'].items()}} for p in out]},
              open(path.replace('.json', '.cert.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
