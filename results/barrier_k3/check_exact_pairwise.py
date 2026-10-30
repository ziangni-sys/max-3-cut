"""Independent exact check of the barrier certificates (Lemma 7.2), by a different method:
for a pair of columns (f,g) write z_s=(f(s),g(s)) in [3]^2 for s in sat_t; then
  slack(f,g) = sum_r nu_r [h_r(f|T_r) != h_r(g|T_r)] - sum_s pi_s [f(s)!=g(s)]
is a sum of unary/pairwise terms in z (since |T_r|<=2); minimise exactly over z in [9]^9."""
import sys, pickle, itertools
from fractions import Fraction as F
from math import lcm
from collections import Counter
import numpy as np

fn, target = sys.argv[1], F(sys.argv[2])
d = pickle.load(open(fn,'rb'))
t, rows = d['t'], d['rows']
sat = [x for x in itertools.product(range(3),repeat=3) if sum(x)%3==t]
assert len(sat)==9
def rat(v):
    q = F(float(v)).limit_denominator(10**6); assert abs(float(q)-float(v))<1e-9; return q
nu = [rat(v) for v in d['nu']]; pi = [rat(v) for v in d['pi']]
assert all(q>=0 for q in nu) and all(q>=0 for q in pi), 'negative weight'
keep = [i for i,q in enumerate(nu) if q>0]
rows = [rows[i] for i in keep]; nu=[nu[i] for i in keep]
D = lcm(*[q.denominator for q in nu+pi if q])
NU = [int(q*D) for q in nu]; PI=[int(q*D) for q in pi]
assert sum(NU)==D, 'sum nu != 1'
print(fn.split('/')[-1], 't',t,'rows kept',len(rows),'D',D,'sum pi',F(sum(PI),D))
assert F(sum(PI),D)==target
for r in rows:
    assert r[0]=='L' and 1<=len(r[1])<=2 and len(r[2])==3**len(r[1])
    assert tuple(r[4])==(0,1,2) and sum(r[3])%3!=t and len(r[3])==3
marg = Counter()
for r,q in zip(rows,nu): marg[tuple(r[3])]+=q
print('  per-tuple x-marginal:', sorted(Counter(str(v) for v in marg.values()).items()), 'support', len(marg))
def cls(x):
    s=len(set(x))
    return 'A' if s==1 else ('D' if s==3 else 'T%d'%(sum(x)%3))
cm = Counter()
for x,v in marg.items(): cm[cls(x)]+=v
print('  class marginal:', {k:str(v) for k,v in sorted(cm.items())})
for x in itertools.product(range(3),repeat=3):
    same = [y for y in itertools.product(range(3),repeat=3) if cls(y)==cls(x)]
    assert marg.get(x,0)==marg.get(same[0],0), 'not uniform within class'
print('  x-marginal uniform within classes: OK')

def colour(r, f):  # f: tuple of 9 values (f(sat[0]),...,f(sat[8]))
    T, h = r[1], r[2]
    return h[sum(f[T[i]]*3**i for i in range(len(T)))]
fixed = ['p1','p2','p3','g0','g1','g2']
def fixcol_row(r, a):
    return r[3][int(a[1])-1] if a[0]=='p' else r[4][int(a[1])]
def fixcol_can(x, a):
    return x[int(a[1])-1] if a[0]=='p' else int(a[1])
worst = None
# fixed-fixed
for a,b in itertools.combinations(fixed,2):
    lhs = sum(n for r,n in zip(rows,NU) if fixcol_row(r,a)!=fixcol_row(r,b))
    rhs = sum(p for x,p in zip(sat,PI) if fixcol_can(x,a)!=fixcol_can(x,b))
    worst = lhs-rhs if worst is None else min(worst,lhs-rhs)
print('  fixed-fixed min slack', worst)
# column-fixed: unary/pairwise in f; just enumerate 3^9
allf = list(itertools.product(range(3),repeat=9))
w2 = None
for a in fixed:
    for f in allf:
        lhs = sum(n for r,n in zip(rows,NU) if colour(r,f)!=fixcol_row(r,a))
        rhs = sum(p for s,(x,p) in enumerate(zip(sat,PI)) if f[s]!=fixcol_can(x,a))
        w2 = lhs-rhs if w2 is None else min(w2,lhs-rhs)
print('  column-fixed min slack', w2)
# column-column via pairwise MRF on z in [9]^9, z = 3*f + g
U = np.zeros((9,9), dtype=np.int64)        # U[s, z_s]
P = np.zeros((9,9,9,9), dtype=np.int64)    # P[s,s', z_s, z_s'] for s<s'
zf = np.arange(9)//3; zg = np.arange(9)%3
for s,p in enumerate(PI): U[s] -= p*(zf!=zg)
for r,n in zip(rows,NU):
    T,h = r[1], np.array(r[2])
    if len(T)==1:
        U[T[0]] += n*(h[zf]!=h[zg])
    else:
        s1,s2 = T
        # colour = h[f(s1)+3 f(s2)]
        cf = h[zf[:,None] + 3*zf[None,:]]; cg = h[zg[:,None]+3*zg[None,:]]
        tab = n*(cf!=cg)   # indexed [z_s1, z_s2]
        if s1<s2: P[s1,s2] += tab
        elif s1>s2: P[s2,s1] += tab.T
        else: raise ValueError('repeated T')
# brute-force minimum over [9]^9: prefix (z0,z1,z2), vectorise z3..z8
sh = (9,)*6
base = np.zeros(sh, dtype=np.int64)
def bc(v, axis):
    shape=[1]*6; shape[axis]=9; return v.reshape(shape)
for s in range(3,9):
    base = base + bc(U[s], s-3)
for s in range(3,9):
    for s2 in range(s+1,9):
        shape=[1]*6; shape[s-3]=9; shape[s2-3]=9
        base = base + P[s,s2].reshape(shape)
m = None
for z0,z1,z2 in itertools.product(range(9),repeat=3):
    const = U[0,z0]+U[1,z1]+U[2,z2]+P[0,1,z0,z1]+P[0,2,z0,z2]+P[1,2,z1,z2]
    add = np.zeros(sh, dtype=np.int64)
    for s in range(3,9):
        v = P[0,s,z0,:]+P[1,s,z1,:]+P[2,s,z2,:]
        add = add + bc(v, s-3)
    val = const + (base+add).min()
    m = val if m is None else min(m,val)
print('  column-column min slack (units 1/D)', m)
assert worst>=0 and w2>=0 and m>=0
print('  EXACT OK: E_nu V_t(x_r,y0) >= %s C_t' % target)
