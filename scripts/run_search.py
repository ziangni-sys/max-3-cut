"""Restartable search for one i-gadget.  Usage: run_search.py k i seed n_outer [max_cols] [init.pkl]"""
import sys, time, pickle, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.driver import Search, SymSearch

k, i, seed, n_outer = map(int, sys.argv[1:5])
max_cols = int(sys.argv[5]) if len(sys.argv) > 5 else 40
init = sys.argv[6] if len(sys.argv) > 6 else f'{_ROOT}/results/reopt_nrt_k{k}.pkl'
if init == 'none':
    cols0 = []
else:
    d = pickle.load(open(init, 'rb'))
    cols0 = dict(d['groups'])[i] if 'groups' in d else d['cols']
ck = f'{_ROOT}/results/search_k{k}_i{i}_s{seed}.ckpt'
def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)
S = SymSearch if os.environ.get("SYM", "1") == "1" else Search
ck = ck.replace(".ckpt", "_sym.ckpt") if S is SymSearch else ck
s = S(k, i, cols0, seed=seed, ckpt=ck, max_cols=max_cols, log=log)
for it in range(n_outer):
    s.outer()
