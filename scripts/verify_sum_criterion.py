"""Exact evaluation of a 3LIN(3) -> MAX-k-CUT gadget family under the sum-over-types criterion (README, "Theorem S"):

    rho = max_{y in Y_k} max_{x in Z_k^3} sum_{t in Z_3} V_t(x, y)  /  sum_t C_t,
    V_t(x, y) = max over auxiliary colourings of I_t,   C_t = min_{x in {0,1,2}^3, x1+x2+x3 = t mod 3} V_t(x, y0).

Self-contained brute force (numpy, integer arithmetic); Y_k = restricted growth strings, y0 = (0, .., k-1).
Also checks the hypothesis of Theorem S that every V_t is symmetric in the two W-primaries (positions 2, 3).
Usage: verify_sum_criterion.py family.json   (format: k, gadgets[{type, n_vertices, edges (1-indexed)}];
vertices 1-3 primaries, 4..3+k globals, rest auxiliary)
"""
import sys, json
from itertools import product
from fractions import Fraction
import numpy as np


def rgs(k):
    out = []
    def rec(p, m):
        if len(p) == k:
            out.append(tuple(p)); return
        for c in range(m + 2 if p else 1):
            rec(p + [c], max(m, c))
    rec([], -1)
    return out


def values(g, k, Y):
    n = g['n_vertices']
    W = np.zeros((n, n), dtype=np.int64)
    for a, b, w in g['edges']:
        if not (isinstance(w, int) and w > 0 and a != b):
            raise ValueError('bad edge')
        W[a - 1, b - 1] += w; W[b - 1, a - 1] += w
    nf = 3 + k
    aux = list(range(nf, n))
    Z = np.array(list(product(range(k), repeat=len(aux))), dtype=np.int64).reshape(k ** len(aux), len(aux))
    Waa = W[np.ix_(aux, aux)]
    aa = np.zeros(len(Z), dtype=np.int64)
    for i in range(len(aux)):
        for j in range(i + 1, len(aux)):
            if Waa[i, j]:
                aa += Waa[i, j] * (Z[:, i] != Z[:, j])
    Wfa = W[:nf, aux]
    V = {}
    for y in Y:
        for x in product(range(k), repeat=3):
            f = np.array(list(x) + list(y))
            ff = sum(int(W[u, v]) for u in range(nf) for v in range(u + 1, nf) if W[u, v] and f[u] != f[v])
            tab = np.array([[int(Wfa[f != c, j].sum()) for c in range(k)] for j in range(len(aux))], dtype=np.int64).reshape(len(aux), k)
            tot = aa + tab[np.arange(len(aux)), Z].sum(axis=1) if len(aux) else aa
            V[x, y] = ff + int(tot.max())
    return V


def main(path):
    d = json.load(open(path))
    k = d['k']
    gs = {g['type']: g for g in d['gadgets']}
    if sorted(gs) != [0, 1, 2]:
        raise ValueError('need one gadget per type 0, 1, 2')
    Y = rgs(k)
    y0 = tuple(range(k))
    V = {t: values(gs[t], k, Y) for t in range(3)}
    for t in range(3):
        for (x, y), v in V[t].items():
            if V[t][(x[0], x[2], x[1]), y] != v:
                raise ValueError(f'type {t} not W-symmetric at x={x} y={y}')
    C = {t: min(V[t][x, y0] for x in product(range(3), repeat=3) if sum(x) % 3 == t) for t in range(3)}
    best, arg = max((sum(V[t][x, y] for t in range(3)), (x, y)) for x in product(range(k), repeat=3) for y in Y)
    rho = Fraction(best, sum(C.values()))
    tw = [sum(w for *_, w in gs[t]['edges']) for t in range(3)]
    print(f'k={k}: C={[C[t] for t in range(3)]} total weights={tw}  max_(x,y) sum_t V_t = {best} at x={arg[0]} y={arg[1]}')
    print(f'rho = {rho} = {float(rho):.6f}   (|Y|={len(Y)}, all x in Z_{k}^3; W-symmetry checked)')
    return rho


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
