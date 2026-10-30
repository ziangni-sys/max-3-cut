import sys
import os, json, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from multiprocessing import Pool
from gadgets.core import edges_to_matrix, allowed_global_assignments
from gadgets.cpsat import cpsat_decide_strong
from gadgets.localsearch import batched_local_search

def job(a):
    W, k, f, thr, h = a
    t0 = time.time()
    try:
        r = cpsat_decide_strong(W, k, list(f), 3 + k, thr, hint=h, time_limit=900.0)
        return ('proved' if r is None else f'violated {r[0]}'), time.time() - t0
    except RuntimeError as e:
        return 'undecided', time.time() - t0

if __name__ == '__main__':
    d = json.load(open(sys.argv[1])); k = d['k']; gi = int(sys.argv[2])
    g = d['gadgets'][gi]; i = g['i']
    W = edges_to_matrix(g['edges'], n=g['n_vertices'])
    Y = allowed_global_assignments(k)
    reps = sorted(set(tuple(sorted(x)) for x in product(range(k), repeat=3)))
    keys = [(x, y) for x in reps for y in Y]
    fixed = np.array([list(x) + list(y) for x, y in keys])
    lv, lc = batched_local_search(W.astype(float), k, fixed, 3 + k, restarts=16, rounds=4)
    thr = {}
    for (x, y), v in zip(keys, lv):
        c = (sum(x) % k == i, y); thr[c] = max(thr.get(c, -1), int(round(v)))
    slack = np.array([thr[(sum(x) % k == i, y)] - v for (x, y), v in zip(keys, lv)])
    order = np.argsort(slack)
    pick = list(order[:4]) + list(order[len(order) // 4: len(order) // 4 + 2]) + list(order[len(order) // 2: len(order) // 2 + 2])
    jobs = [(W, k, fixed[j], thr[(sum(keys[j][0]) % k == i, keys[j][1])], lc[j]) for j in pick]
    with Pool(4) as p:
        res = p.map(job, jobs, chunksize=1)
    for j, r in zip(pick, res):
        print(keys[j], 'slack', slack[j], r, flush=True)
