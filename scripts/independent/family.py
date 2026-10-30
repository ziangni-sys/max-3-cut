"""Independent rebuild of the family (2A_L, N_L, N_L) from the text of Section 5, and exact evaluation.
Vertices: p1,p2,p3 -> 0,1,2 ; globals g_0..g_{k-1} -> 3..3+k-1 ; aux after.
Universal u_i = g_{2+i}, i=1..L.
"""
import sys, itertools
import numpy as np
from fractions import Fraction as F
from math import comb

E_EDGES = [('p1','a1'),('p1','a2'),('p1','a3'),('p1','a4'),('p2','a3'),('p2','a4'),('p2','a5'),('p2','a6'),
           ('p3','a1'),('p3','a2'),('p3','a5'),('p3','a6'),
           ('a1','a6'),('a1','a4'),('a2','a3'),('a2','a5'),('a3','a6'),('a4','a5')]

def idx(name, k):
    if name[0]=='p': return int(name[1])-1
    if name[0]=='g': return 3+int(name[1:])
    if name[0]=='u': return 3+2+int(name[1:])   # u_i = g_{2+i}
    raise ValueError

def build_A(L):
    k = 3+L
    edges = {}
    def add(a,b,w):
        key = (a,b)
        edges[key] = edges.get(key,0)+w
    for a,b in E_EDGES: add(a,b,1)
    for j in range(1,7):
        for i in range(1,L+1): add('a%d'%j,'u%d'%i,1)
    for i in range(1,L+1):
        for i2 in range(i+1,L+1): add('u%d'%i,'u%d'%i2,3)
    return edges

def build_N(L):
    edges = {}
    def add(a,b,w): edges[(a,b)] = edges.get((a,b),0)+w
    add('p1','p2',1); add('p1','p3',1); add('p2','p3',1)
    for j in (1,2,3): add('b','p%d'%j,3)
    for i in range(1,L+1): add('b','u%d'%i,3)
    for j in (1,2,3):
        for i in range(1,L+1): add('p%d'%j,'u%d'%i,3)
    return edges

def scale(edges, c): return {e: c*w for e,w in edges.items()}

def value_fn(edges, k):
    """return function V(x, y) maximizing over aux colourings (exhaustive)."""
    aux = sorted({v for e in edges for v in e if v[0] in 'ab'})
    na = len(aux)
    aidx = {a:i for i,a in enumerate(aux)}
    AC = np.array(list(itertools.product(range(k), repeat=na)), dtype=np.int64)  # all aux colourings
    auxaux = np.zeros(len(AC), dtype=np.int64)
    fixed_aux = []   # (aux index, fixed name, w)
    fixed_fixed = []
    for (a,b),w in edges.items():
        ia, ib = a in aidx, b in aidx
        if ia and ib: auxaux += w*(AC[:,aidx[a]] != AC[:,aidx[b]])
        elif ia: fixed_aux.append((aidx[a], b, w))
        elif ib: fixed_aux.append((aidx[b], a, w))
        else: fixed_fixed.append((a,b,w))
    def colour_of(name, x, y):
        if name[0]=='p': return x[int(name[1])-1]
        if name[0]=='u': return y[2+int(name[1:])]
        if name[0]=='g': return y[int(name[1:])]
    cols = np.arange(k)
    def V(x, y):
        tot = sum(w*(colour_of(a,x,y)!=colour_of(b,x,y)) for a,b,w in fixed_fixed)
        gain = np.zeros((na, k), dtype=np.int64)
        for j, f, w in fixed_aux:
            gain[j] += w*(cols != colour_of(f,x,y))
        if na == 0: return tot
        s = auxaux.copy()
        for j in range(na): s += gain[j][AC[:,j]]
        return tot + int(s.max())
    return V

def rgs(k):
    out = []
    def rec(pref, m):
        if len(pref)==k: out.append(tuple(pref)); return
        for c in range(m+2 if pref else 1):
            if c < k: rec(pref+[c], max(m,c))
    rec([], -1)
    return out

def weight(edges): return sum(edges.values())

if __name__ == '__main__':
    L = int(sys.argv[1])
    k = 3+L
    A = scale(build_A(L), 2); N = build_N(L)
    print('L',L,'k',k,'w(2A)',weight(A),'w(N)',weight(N), 'formula', 2*(18+6*L+3*comb(L,2)), 12+12*L)
    VA, VN = value_fn(A,k), value_fn(N,k)
    Ys = rgs(k)
    print('|Y_k|', len(Ys))
    y0 = tuple(range(k))
    C = []
    for t in range(3):
        V = VA if t==0 else VN
        C.append(min(V(x,y0) for x in itertools.product(range(3),repeat=3) if sum(x)%3==t))
    print('C', C, 'sumC', sum(C))
    best = -1; argbest = []
    sym_ok = True
    table = {}
    for y in Ys:
        for x in itertools.product(range(k), repeat=3):
            a = VA(x,y); n = VN(x,y)
            # symmetry check under all permutations of primaries
            for perm in itertools.permutations(range(3)):
                xp = tuple(x[p] for p in perm)
                if VA(xp,y)!=a or VN(xp,y)!=n: sym_ok=False
            s = a + 2*n
            table[(x,y)] = (a,n,s)
            if s > best: best = s; argbest=[(x,y)]
            elif s == best: argbest.append((x,y))
    print('max sum V', best, 'rho', F(best, sum(C)), 'sym', sym_ok)
    print('formula 1-2/(29+18L+1.5L(L-1)) =', 1-F(2)/(29+18*L+F(3,2)*L*(L-1)))
    const = [(x,y) for (x,y) in argbest if x[0]==x[1]==x[2]]
    print('#argmax', len(argbest), 'constant-x argmax e.g.', const[:3])
    import pickle; pickle.dump((C,table), open('fam_L%d.pkl'%L,'wb'))
