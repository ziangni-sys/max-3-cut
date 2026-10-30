"""The unified family of the paper for MAX-(3+L)-CUT, L = 0, 1, 2, ... (k = 3 + L colours).

Vertices (1-indexed): 1-3 primaries, 4..3+k globals; the universal globals are U = {7, .., 6+L} (colours 3..k-1 at y0).
  A_L : NRT's 18-edge type-0 gadget E on auxiliaries a1..a6 (unit weights), every a_j joined to every u in U
        with weight 1, and every pair of U joined with weight 3.
  N_L : unit triangle on the primaries; one auxiliary b joined to x1, x2, x3 and every u in U with weight 3;
        every primary joined to every u in U with weight 3.
Type 0 gets 2*A_L, types 1 and 2 get N_L.  L = 0 is the 27/29 family (2E, N, N).
Usage: make_unified_family.py L out.json
"""
import sys
import os, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from gadgets.nrt_gadgets import NRT_K3_G0


def family(L):
    k = 3 + L
    U = list(range(7, 7 + L))
    base = 3 + k                                   # last global index (1-indexed)
    # A_L: relabel NRT's aux 7..12 to base+1..base+6
    amap = {v: base + v - 6 for v in range(7, 13)}
    A = [(a if a <= 3 else amap[a], b if b <= 3 else amap[b], 1) for a, b, _ in NRT_K3_G0]
    A += [(amap[v], u, 1) for v in range(7, 13) for u in U]
    A += [(u, v, 3) for i, u in enumerate(U) for v in U[i + 1:]]
    A = [(a, b, 2 * w) for a, b, w in A]           # type 0 gets 2*A_L
    b = base + 1
    N = [(1, 2, 1), (1, 3, 1), (2, 3, 1)] + [(j, b, 3) for j in (1, 2, 3)] + [(u, b, 3) for u in U]
    N += [(j, u, 3) for j in (1, 2, 3) for u in U]
    return {'k': k, 'L': L, 'construction': 'make_unified_family.py',
            'gadgets': [{'type': 0, 'n_vertices': base + 6, 'edges': [list(e) for e in A]},
                        {'type': 1, 'n_vertices': base + 1, 'edges': [list(e) for e in N]},
                        {'type': 2, 'n_vertices': base + 1, 'edges': [list(e) for e in N]}]}


if __name__ == '__main__':
    json.dump(family(int(sys.argv[1])), open(sys.argv[2], 'w'), indent=1)
