import sys, time, pickle, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product
from multiprocessing import Pool
from gadgets.core import allowed_global_assignments
from gadgets.cpsat import cpsat_decide_strong
from gadgets.localsearch import batched_local_search
def job(a):
    W, k, f, thr, h = a
    t0 = time.time()
    try:
        import os
        r = cpsat_decide_strong(W, k, list(f), 3 + k, thr, hint=h, time_limit=300.0,
                                value_precedence=os.environ.get('VP') == '1', workers=int(os.environ.get('WK', '1')))
        return ('proved' if r is None else f'violated'), round(time.time() - t0, 1)
    except RuntimeError:
        return 'undecided', round(time.time() - t0, 1)
if __name__ == '__main__':
    st = pickle.load(open(sys.argv[1], 'rb')); k = st['k']; i = st['i']
    Wf = st['best']['W']; W = np.rint(Wf / Wf.max() * 1e6).astype(np.int64)
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
    if os.environ.get('RANDOM_SAMPLE'):
        pick = list(np.random.default_rng(7).choice(len(keys), int(os.environ['RANDOM_SAMPLE']), replace=False))
    else:
        pick = list(order[:6]) + list(order[len(order)//2:len(order)//2+2])
    import os
    with Pool(int(os.environ.get('NP', '4'))) as p:
        res = p.map(job, [(W, k, fixed[j], thr[(sum(keys[j][0]) % k == i, keys[j][1])], lc[j]) for j in pick], chunksize=1)
    print('n_aux', W.shape[0] - 3 - k)
    for j, r in zip(pick, res): print(keys[j], 'slack', slack[j], r)
