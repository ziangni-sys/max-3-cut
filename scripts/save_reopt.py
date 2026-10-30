import sys
import os, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from gadgets.search import *
from gadgets.io import save_family
k = int(sys.argv[1]); scale = int(sys.argv[2])
d = pickle.load(open(f'{_ROOT}/results/reopt_nrt_k{k}.pkl', 'rb'))
groups = [Group(k, i, cols) for i, cols in d['groups']]
class Dummy: pass
lp = Dummy(); lp.groups = groups; lp.k = k
m = max(w.max() for w in d['w'])
iw = [integerize(w, scale, m) for w in d['w']]
r = save_family(f'{_ROOT}/results/k{k}_reopt_nrt_cols_s{scale}.json', k, [full_gadget_matrices(lp, iw)[i] for i in range(k)],
                note='LP re-optimization of NRT witness columns, independent gadget per predicate')
print(r, float(r))
