import sys
import os, json, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np
from gadgets.core import edges_to_matrix
from gadgets.sdpbound import sdp_upper_bound
d = json.load(open('results/fam_k5_c32.json')); k = 5
g = d['gadgets'][0]
W = edges_to_matrix(g['edges'], n=g['n_vertices'])
cache = [json.loads(l) for l in open('results/fam_k5_c32.cache.jsonl')]
cache.sort(key=lambda e: -e['secs'])
for e in cache[:10] + cache[40:45]:
    t0 = time.time()
    ub, st = sdp_upper_bound(W, k, e['x'] + e['y'], 3 + k)
    print(e['x'], e['y'], 'cpsat', e['secs'], 's  threshold', e['proved_max'], ' SDP ub', round(ub), st,
          ' certifies' if ub <= e['proved_max'] * (1 - 1e-7) else ' NO', f'{time.time()-t0:.1f}s', flush=True)
