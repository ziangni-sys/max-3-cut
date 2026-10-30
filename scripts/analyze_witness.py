"""Which functions of the satisfying assignment do NRT's auxiliary vertices compute in their witnesses?"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from itertools import product, permutations
from gadgets.core import *
from gadgets.nrt_gadgets import *

def all_argmax(solver, x, y):
    col_fixed = list(x) + list(y)
    fA = (solver.pairA + solver._unary(solver.A, solver.ZA, col_fixed)).astype(float)
    fB = (solver.pairB + solver._unary(solver.B, solver.ZB, col_fixed)).astype(float)
    tot = fA[:, None] + fB[None, :] + solver.cross
    m = tot.max()
    ia, ib = np.nonzero(tot >= m - 1e-9)
    Z = np.concatenate([solver.ZA[ia], solver.ZB[ib]], axis=1)
    return Z  # rows: optimal aux colorings (aux order = solver.A + solver.B)

def describe(k, W, i, label):
    s = FixedMaxCut(W, k)
    sat = sat_assignments(k, i)
    y0 = tuple(range(k))
    opts = [all_argmax(s, x, y0) for x in sat]
    print(label, 'num optimal witnesses per sat x:', [len(o) for o in opts])
    aux = s.A + s.B
    # for each aux vertex: the set of possible values at each sat x
    for a, u in enumerate(aux):
        vals = [sorted(set(o[:, a].tolist())) for o in opts]
        # test affine fits a1 x1 + a2 x2 + a3 x3 + c (mod k), over sat x, allowing any value in vals
        fits = []
        for a1, a2, a3, c in product(range(k), repeat=4):
            if all(((a1*x[0]+a2*x[1]+a3*x[2]+c) % k) in v for x, v in zip(sat, vals)):
                fits.append((a1, a2, a3, c))
        print(f'  aux v{u+1}: deg={int((W[u]>0).sum())} values-per-x={["".join(map(str,v)) for v in vals]} affine fits={fits[:4]}{"..." if len(fits)>4 else ""}')

describe(3, edges_to_matrix(NRT_K3_G0, n=14), 0, 'k=3 I0')
describe(3, edges_to_matrix(NRT_K3_G1, n=14), 1, 'k=3 I1')
describe(4, edges_to_matrix(NRT_K4_G0, n=19), 0, 'k=4 I0')
