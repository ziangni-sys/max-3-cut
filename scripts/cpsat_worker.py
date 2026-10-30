"""Exact max over auxiliary colorings for a batch of fixed colorings (separate process:
OR-tools and highspy clash).  Usage: cpsat_worker.py in.npz out.npz workers"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from multiprocessing import Pool
from gadgets.cpsat import cpsat_max, cpsat_decide_strong

def job(args):
    W, k, fixed, hint, thr = args
    if thr is None:
        v, col = cpsat_max(W, k, list(fixed), 3 + k, time_limit=3600, hint=hint)
        return v, col
    r = cpsat_decide_strong(W, k, list(fixed), 3 + k, int(thr), hint=hint, time_limit=6 * 3600.0)
    return (None, None) if r is None else r

if __name__ == '__main__':
    d = np.load(sys.argv[1])
    W, k, F, H = d['W'], int(d['k']), d['fixed'], d['hints']
    T = d['thr'] if 'thr' in d.files else [None] * len(F)
    nw = int(sys.argv[3])
    with Pool(nw) as p:
        res = p.map(job, [(W, k, f, h, (None if t is None or t < 0 else t)) for f, h, t in zip(F, H, T)], chunksize=1)
    found = np.array([r[0] is not None for r in res])
    cols = np.array([r[1] if r[1] is not None else list(f) + [0] * (W.shape[0] - len(f)) for r, f in zip(res, F)])
    np.savez(sys.argv[2], found=found, cols=cols)
