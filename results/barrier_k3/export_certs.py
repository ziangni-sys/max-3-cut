"""Export the barrier certificates as exact integer JSON files (cert_t{0,1,2}.json).
Each row: x (unsatisfying tuple), T (1 or 2 satisfying tuples, as indices into sat_t in lexicographic order),
h (table [3]^|T| -> [3], lexicographic), NU (integer weight).  PI: integer weight per satisfying tuple.
nu = NU/D, pi = PI/D.  A column f gets colour h(f|T) in the row (f|T in lexicographic index order)."""
import pickle, json
from fractions import Fraction as F
from math import lcm
from itertools import product
for t, fn in [(0, 'mudual_t0_0_0_9_9.pkl'), (1, 'mudual_t1_5_6_0_9.pkl'), (2, 'mudual_t2_mirror.pkl')]:
    d = pickle.load(open(fn, 'rb'))
    rat = lambda v: F(float(v)).limit_denominator(10 ** 6)
    nu = [rat(v) for v in d['nu']]; pi = [rat(v) for v in d['pi']]
    keep = [i for i, q in enumerate(nu) if q > 0]
    D = lcm(*[q.denominator for q in [nu[i] for i in keep] + pi if q != 0])
    sat = [x for x in product(range(3), repeat=3) if sum(x) % 3 == t]
    rows = []
    for i in keep:
        _, T, h, x, y = d['rows'][i]
        assert tuple(y) == (0, 1, 2)
        rows.append({'x': list(x), 'T': [int(v) for v in T], 'h': [int(v) for v in h], 'NU': int(nu[i] * D)})
    out = {'type': t, 'D': D, 'sat_t': [list(x) for x in sat], 'PI': [int(q * D) for q in pi],
           'target_sum_pi': str(F(sum(int(q * D) for q in pi), D)), 'rows': rows,
           'format': __doc__}
    json.dump(out, open(f'cert_t{t}.json', 'w'), indent=0)
    print(t, 'D', D, 'rows', len(rows), 'sum pi', out['target_sum_pi'], 'sum NU', sum(r['NU'] for r in rows))
