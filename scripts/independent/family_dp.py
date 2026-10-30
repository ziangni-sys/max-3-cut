"""Pattern-based evaluation of (2A_L,N_L,N_L) for larger L, A_L's aux optimisation by DP on the 6-cycle."""
import sys, itertools, pickle
import numpy as np
from fractions import Fraction as F
from math import comb
from family import E_EDGES

# aux class: which primaries each aux is adjacent to
adjp = {j: [] for j in range(1,7)}
for a,b in E_EDGES:
    if a[0]=='p' and b[0]=='a': adjp[int(b[1])].append(int(a[1])-1)
cyc_edges = [(int(a[1]),int(b[1])) for a,b in E_EDGES if a[0]=='a']
# order the 6-cycle
nbr = {j: [] for j in range(1,7)}
for a,b in cyc_edges: nbr[a].append(b); nbr[b].append(a)
order=[1]; prev=None
while len(order)<6:
    cur=order[-1]; nxt=[v for v in nbr[cur] if v!=prev and v not in order][0]; prev=cur; order.append(nxt)
assert order[0] in nbr[order[-1]]

def VA(x, u, k):   # weight-1 A_L (not doubled)
    L = len(u)
    tot = 3*sum(u[i]!=u[j] for i in range(L) for j in range(i+1,L))
    gain = {}
    for j in range(1,7):
        g = np.zeros(k, dtype=np.int64)
        for p in adjp[j]: g += (np.arange(k)!=x[p])
        for ui in u: g += (np.arange(k)!=ui)
        gain[j]=g
    # cycle DP: maximize sum gain + sum_{cycle edges}[c!=c']
    D = 1-np.eye(k, dtype=np.int64)
    best = -10**9
    for c0 in range(k):
        val = np.full(k, -10**9, dtype=np.int64); val[c0] = gain[order[0]][c0]
        for v in order[1:]:
            val = (val[:,None] + D).max(0) + gain[v]
        best = max(best, int((val + D[:,c0]).max()))
    return tot + best

def VN(x, u, k):
    tot = sum(x[i]!=x[j] for i in range(3) for j in range(i+1,3))
    tot += 3*sum(xj!=ui for xj in x for ui in u)
    nb = list(x)+list(u)
    tot += 3*max(sum(c!=v for v in nb) for c in range(k))
    return tot

def setparts(n):
    # restricted growth strings of length n
    out=[]
    def rec(p,m):
        if len(p)==n: out.append(tuple(p)); return
        for c in range(m+2): rec(p+[c], max(m,c))
    rec([0],0) if n>0 else None
    return out

if __name__=='__main__':
    for L in range(0, int(sys.argv[1])+1):
        k=3+L
        C0 = min(2*VA(x, tuple(range(3,k)), k) for x in itertools.product(range(3),repeat=3) if sum(x)%3==0)
        C1 = min(VN(x, tuple(range(3,k)), k) for x in itertools.product(range(3),repeat=3) if sum(x)%3==1)
        C2 = min(VN(x, tuple(range(3,k)), k) for x in itertools.product(range(3),repeat=3) if sum(x)%3==2)
        best=-1; arg=None; cnt=0
        for pat in setparts(3+L):
            if max(pat) >= k: continue
            x, u = pat[:3], pat[3:]
            s = 2*VA(x,u,k) + 2*VN(x,u,k); cnt+=1
            if s>best: best=s; arg=pat
        SC=C0+C1+C2
        print(f'L={L} k={k} C=({C0},{C1},{C2}) sumC={SC} max={best} at {arg} rho={F(best,SC)} formula={1-F(2)/(29+18*L+F(3,2)*L*(L-1))} patterns={cnt} sumC-max={SC-best}')
