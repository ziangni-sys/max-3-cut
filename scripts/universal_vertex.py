"""MAX-3-CUT -> MAX-(3+L)-CUT by adding L universal vertices.

H = G + u_1..u_L, edge (u_i, v) of weight d_v/3 for every v in G (d_v = weighted degree), and for L >= 2 edges
(u_i, u_j) of weight w.  Lemma: min(w, uncut_3(G)) <= min uncut of a (3+L)-colouring of H <= uncut_3(G)
(equality with uncut_3(G) when w >= uncut_3(G); for L = 1 always).  If two u's share a colour we lose >= w;
otherwise a vertex v coloured like some u_i loses d_v/3 on that edge and can be recoloured with the best of the 3
colours not used by the u's at cost <= d_v/3.  This is the reduction of Kann-Khanna-Lagergren-Panconesi (1997, L=1)
and Guruswami-Sinop (2009, App. D; L <= 2); what is new here is only feeding in the K = 3 base constant.  Total weight of H:
W = M (1 + 2L/3) + C(L,2) w.  From a MAX-3-CUT gap with uncut fractions delta_c (YES) / delta_s (NO), w = delta_s M:
  ratio = (W/M - delta_s) / (W/M - delta_c).
Part 1 brute-forces the lemma on random weighted graphs (exact rationals); part 2 tabulates the implied bounds
(also trying Guruswami-Sinop's tensor reduction to K = 3q colours as the base).
"""
import itertools, random, sys
from fractions import Fraction as F
from math import comb


def uncut(n, edges, k):
    best = None
    for col in itertools.product(range(k), repeat=n - 1):
        col = (0,) + col
        u = sum(w for a, b, w in edges if col[a] == col[b])
        best = u if best is None or u < best else best
    return best


def check(trials, seed=7):
    rng = random.Random(seed)
    for _ in range(trials):
        n = rng.randint(3, 6)
        edges = [(a, b, F(rng.randint(1, 5))) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.7]
        if not edges:
            continue
        deg = [sum(w for a, b, w in edges if v in (a, b)) for v in range(n)]
        u3 = uncut(n, edges, 3)
        for L in (1, 2, 3):
            H = list(edges)
            for u in range(n, n + L):
                H += [(v, u, deg[v] / 3) for v in range(n) if deg[v] > 0]
            H += [(a, b, max(u3, F(1))) for a, b in itertools.combinations(range(n, n + L), 2)]
            assert uncut(n + L, H, 3 + L) == u3
    print(f'lemma verified on {trials} random weighted graphs, L = 1, 2, 3')


def best_ratio(k, dc, ds):
    out = []
    for K in range(3, k + 1, 3):
        if K == 3:
            WK, uc, us = F(1), dc, ds
        else:                                    # GS09 tensor reduction 3 -> K, exact total weight m(K^2 - 2K/3)
            WK, uc, us = K * K - F(2 * K, 3), K * dc, K * ds
        L = k - K
        W = WK * (1 + F(2 * L, K)) + comb(L, 2) * us
        out.append(((W - us) / (W - uc), K, L))
    return min(out)


if __name__ == '__main__':
    check(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
    print(' k | stated in literature | from AOTW 16/17 (perfect completeness) | from (29/30, 9/10)')
    for k in range(3, 13):
        c = k % 3
        lit = F(16, 17) if k == 3 else 1 - F(1, 17 * (k + c) + c)
        a, b = best_ratio(k, F(0), F(1, 17)), best_ratio(k, F(1, 30), F(1, 10))
        print(f'{k:2d} | {str(lit):>8} = {float(lit):.6f} | {str(a[0]):>8} = {float(a[0]):.6f} | {str(b[0]):>8} = {float(b[0]):.6f}')
