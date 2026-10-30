"""Shared-global refinement: per-y soundness parameters and the refined ratio
   b'/a with b' = max_{y in Y} [ sum_i S2_i(y) + max_r (S1_r(y) - S2_r(y)) ]."""
import sys
import os, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from fractions import Fraction
from gadgets.core import *

def per_y_params(W, k, i):
    fm = FixedMaxCut(W, k)
    Y = allowed_global_assignments(k)
    out = {}
    for y in Y:
        s1 = max(fm.maxval(x, y) for x in sat_assignments(k, i))
        s2 = max(fm.maxval(x, y) for x in unsat_assignments(k, i))
        out[y] = (s1, s2)
    C = min(fm.maxval(x, tuple(range(k))) for x in sat_assignments(k, i))
    return C, out

def refined_ratio(Cs, pys, k):
    Y = allowed_global_assignments(k)
    best = None
    for y in Y:
        val = sum(p[y][1] for p in pys) + max(0, max(p[y][0] - p[y][1] for p in pys))
        if best is None or val > best[0]:
            best = (val, y)
    return Fraction(best[0], sum(Cs)), best[1]

if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    k = d['k']
    Cs, pys = [], []
    for g in d['gadgets']:
        W = edges_to_matrix(g['edges'], n=g['n_vertices'])
        C, py = per_y_params(W, k, g['i'])
        Cs.append(C); pys.append(py)
        print('i', g['i'], 'C', C, {''.join(map(str, y)): v for y, v in py.items()})
    r, y = refined_ratio(Cs, pys, k)
    print('original ratio', d['ratio_float'], ' refined (shared y) ratio', r, float(r), 'worst y', y)
