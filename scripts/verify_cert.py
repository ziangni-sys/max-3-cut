"""Certificate-style exact verification of a 3LIN(k)->MAX-k-CUT gadget family.

* Symmetry (optional, checked): for each gadget, vertex permutations P that fix the globals,
  permute the primaries, and satisfy W[P u, P v] = W[u, v] exactly.  If they realize all of
  S_3 on the primaries, then max_z I(x,y,z) depends on x only up to permutation, so only
  sorted x are examined.
* Lower bounds: explicit colorings (local search).  Upper bounds: CP-SAT proves that no
  auxiliary coloring beats the threshold (status INFEASIBLE); a found violation raises the
  threshold to the proven optimum of that instance.
* C is certified from below by explicit witnesses at y0 (conservative for the ratio).
Reports the NRT Thm A.4 ratio and the shared-y refined ratio, both from certified bounds.
Usage: verify_cert.py family.json [procs]
"""
import sys
import os, json, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from fractions import Fraction
from itertools import product, permutations
from multiprocessing import Pool
import numpy as np
from gadgets.core import edges_to_matrix, allowed_global_assignments
from gadgets.cpsat import cpsat_decide, cpsat_max
from gadgets.localsearch import batched_local_search


def decide_job(a, workers=1, time_limit=1800.0):
    W, k, fixed, thr, hint = a
    try:
        r = cpsat_decide(W, k, list(fixed), 3 + k, thr, hint=hint, workers=workers, time_limit=time_limit)
        if r is None:
            return None
        v, col = cpsat_max(W, k, list(fixed), 3 + k, time_limit=4 * time_limit, hint=r[1], workers=workers)
        return v, col
    except RuntimeError:
        return 'UNDECIDED'


def check_automorphisms(W, k, perms):
    n = W.shape[0]
    realized = set()
    for P in perms:
        P = [p - 1 for p in P]
        assert sorted(P) == list(range(n)), 'not a permutation'
        assert all(P[3 + j] == 3 + j for j in range(k)), 'globals must be fixed'
        assert sorted(P[:3]) == [0, 1, 2], 'primaries must map to primaries'
        Pa = np.array(P)
        assert np.array_equal(W[np.ix_(Pa, Pa)], W), 'not an automorphism'
        realized.add(tuple(P[:3]))
    return realized


def main():
    path = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    d = json.load(open(path))
    k = d['k']
    Y = allowed_global_assignments(k)
    y0 = tuple(range(k))
    out = []
    for g in d['gadgets']:
        t0 = time.time()
        i = g['i']
        W = edges_to_matrix(g['edges'], n=g['n_vertices'])
        sym = False
        if g.get('automorphisms'):
            realized = check_automorphisms(W, k, g['automorphisms'])
            sym = realized == set(permutations(range(3)))
        X = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3))) if sym \
            else list(product(range(k), repeat=3))
        keys = [(x, y) for x in X for y in Y]
        fixed = np.array([list(x) + list(y) for x, y in keys])
        lv, lc = batched_local_search(W.astype(float), k, fixed, 3 + k, restarts=24, rounds=5)
        lv = np.rint(lv).astype(np.int64)
        # C from explicit witnesses at y0, over ALL satisfying x (no symmetry needed)
        satx = [x for x in product(range(k), repeat=3) if sum(x) % k == i]
        fx = np.array([list(x) + list(y0) for x in satx])
        cv, _ = batched_local_search(W.astype(float), k, fx, 3 + k, restarts=24, rounds=5)
        C = int(np.rint(cv.min()))
        thr = {}
        for (x, y), v in zip(keys, lv):
            key = (sum(x) % k == i, y)
            thr[key] = max(thr.get(key, -1), int(v))
        jobs = [(W, k, f, thr[(sum(x) % k == i, y)], h) for (x, y), f, h in zip(keys, fixed, lc)]
        with Pool(procs) as p:
            res = p.map(decide_job, jobs, chunksize=1)
        for j, r in enumerate(res):           # hard instances: retry with all cores, longer
            if isinstance(r, str):
                print(f'   retrying hard instance {keys[j]} with {procs} workers', flush=True)
                r = decide_job(jobs[j], workers=procs, time_limit=4 * 3600.0)
                if isinstance(r, str):
                    raise RuntimeError(f'instance {keys[j]} undecided')
                res[j] = r
        viol = 0
        for (x, y), r in zip(keys, res):
            if r is not None:
                viol += 1
                key = (sum(x) % k == i, y)
                thr[key] = max(thr[key], r[0])
        S1y = {y: thr[(True, y)] for y in Y}
        S2y = {y: thr[(False, y)] for y in Y}
        t = int(np.triu(W, 1).sum())
        out.append(dict(i=i, C=C, S1y=S1y, S2y=S2y, S1=max(S1y.values()), S2=max(S2y.values()), t=t))
        print(f'i={i}: C>={C} S1<={max(S1y.values())} S2<={max(S2y.values())} t={t} n_aux={W.shape[0]-3-k} '
              f'sym={sym} instances={len(keys)} local-search-misses={viol} ({time.time()-t0:.0f}s)', flush=True)
    sumC = sum(p['C'] for p in out)
    nrt = Fraction(sum(p['S2'] for p in out) + max(p['S1'] - p['S2'] for p in out), sumC)
    ref = Fraction(max(sum(p['S2y'][y] for p in out) + max(0, max(p['S1y'][y] - p['S2y'][y] for p in out))
                       for y in Y), sumC)
    print(f'certified NRT Thm A.4 ratio      <= {nrt} = {float(nrt):.7f}')
    print(f'certified shared-y refined ratio <= {ref} = {float(ref):.7f}')
    json.dump({'nrt_ratio': [nrt.numerator, nrt.denominator], 'refined_ratio': [ref.numerator, ref.denominator],
               'gadgets': [{'i': p['i'], 'C_lower': p['C'], 'S1_upper': p['S1'], 'S2_upper': p['S2'], 't': p['t'],
                            'S1y': {''.join(map(str, y)): v for y, v in p['S1y'].items()},
                            'S2y': {''.join(map(str, y)): v for y, v in p['S2y'].items()}} for p in out]},
              open(path.replace('.json', '.cert.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
