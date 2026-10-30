"""Save / load gadget families as JSON edge lists (1-indexed, NRT format)."""
import json
import numpy as np
from .core import gadget_params, hardness_ratio
from .search import prune_isolated


def matrix_to_edges(W):
    n = W.shape[0]
    return [(u + 1, v + 1, int(W[u, v])) for u in range(n) for v in range(u + 1, n) if W[u, v]]


def save_family(path, k, mats, note='', params=None):
    mats = [prune_isolated(np.asarray(W, dtype=np.int64), k) for W in mats]
    if params is None:
        params = [gadget_params(W, k, i) for i, W in enumerate(mats)]
    r = hardness_ratio(params)
    json.dump({'k': k, 'note': note,
               'ratio': [r.numerator, r.denominator], 'ratio_float': float(r),
               'gadgets': [{'i': i, 'n_vertices': int(W.shape[0]), 'edges': matrix_to_edges(W), 'params': p}
                           for i, (W, p) in enumerate(zip(mats, params))]},
              open(path, 'w'), indent=1)
    return r


def load_family(path):
    d = json.load(open(path))
    from .core import edges_to_matrix
    mats = [edges_to_matrix(g['edges'], n=g['n_vertices']) for g in d['gadgets']]
    return d['k'], mats, d
