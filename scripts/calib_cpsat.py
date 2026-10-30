import sys
import os, time, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
from gadgets.core import *
from gadgets.core import edges_to_matrix
import json
def load_family(p):
    d = json.load(open(p)); return d["k"], [edges_to_matrix(g["edges"], n=g["n_vertices"]) for g in d["gadgets"]], d
k, mats, d = load_family(_ROOT + '/results/k4_reopt_nrt_cols_s20000.json')
t0 = time.time()
p = gadget_params_cpsat(mats[0], k, 0)
print('cpsat', p, f'{time.time()-t0:.0f}s', 'claimed', d['gadgets'][0]['params'])
