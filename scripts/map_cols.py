"""Map columns of an i-gadget to a (u*i)-gadget via x -> u*x, colours c -> u*c (u a unit mod k).
Usage: map_cols.py k i u in.pkl out.pkl"""
import sys
import os, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from gadgets.core import sat_assignments
k, i, u = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
st = pickle.load(open(sys.argv[4], 'rb'))
cols = st['best']['cols'] if 'best' in st and st['best'].get('cols') else st['cols']
uinv = next(v for v in range(k) if (u * v) % k == 1)
j = (u * i) % k
src = {x: n for n, x in enumerate(sat_assignments(k, i))}
dst = sat_assignments(k, j)
out = []
for f in cols:
    out.append(tuple((u * f[src[tuple((uinv * v) % k for v in xp)]]) % k for xp in dst))
pickle.dump({'cols': out, 'g': None}, open(sys.argv[5], 'wb'))
print(f'mapped {len(out)} columns from i={i} to i={j}')
