"""Reproduce the gadget parameters of NRT (arXiv:2509.18057, Lemmas A.5, A.6)."""
import sys
import os, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.core import *
from gadgets.nrt_gadgets import NRT_K3_G0, NRT_K3_G1, NRT_K4_G0

def run(k, mats, label):
    ps = []
    for i, W in enumerate(mats):
        t0 = time.time()
        p = gadget_params(W, k, i)
        ps.append(p)
        print(f'{label} i={i}: {p}  ({time.time()-t0:.1f}s)', flush=True)
    r = hardness_ratio(ps)
    print(f'{label}: ratio = {r} = {float(r):.6f}\n', flush=True)
    return ps, r

# k = 3
n3 = 14
G0 = edges_to_matrix(NRT_K3_G0, n=n3)
G1 = edges_to_matrix(NRT_K3_G1, n=n3)
G2 = relabel_globals(G1, 3, [0, 2, 1])      # (4,5,6) -> (4,6,5)
run(3, [G0, G1, G2], 'k=3')

# k = 4
which = sys.argv[1] if len(sys.argv) > 1 else 'a'
H0 = edges_to_matrix(NRT_K4_G0, n=19)
perm = [3, 0, 1, 2] if which == 'a' else [1, 2, 3, 0]
Hs = [H0]
for i in range(1, 4):
    Hs.append(relabel_globals(Hs[-1], 4, perm))
run(4, Hs, f'k=4 perm={perm}')
