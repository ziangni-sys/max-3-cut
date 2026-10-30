"""Batched local search for max over auxiliary colorings (numpy only)."""
import numpy as np


def batched_local_search(W, k, fixed, n_fixed, restarts=8, rounds=3, rng=None, init=None):
    """W: (n,n) float.  fixed: (B, n_fixed) colors of the first n_fixed vertices.
    Returns best values (B,) and colorings (B, n)."""
    rng = np.random.default_rng() if rng is None else rng
    B = fixed.shape[0]
    n = W.shape[0]
    R = restarts
    col = np.empty((B * R, n), dtype=np.int64)
    col[:, :n_fixed] = np.repeat(fixed, R, axis=0)
    col[:, n_fixed:] = rng.integers(0, k, size=(B * R, n - n_fixed))
    if init is not None:   # seed the first restart of each item with a given coloring
        col[::R, n_fixed:] = init[:, n_fixed:]
    aux = np.arange(n_fixed, n)
    totW = W.sum() / 2
    if len(aux) == 0:
        c = np.asarray(fixed)
        vals = np.array([(W * (r[:, None] != r[None, :])).sum() / 2 for r in c])
        return vals, c.copy()

    def field(col):
        O = np.zeros((col.shape[0], n, k))
        np.put_along_axis(O, col[:, :, None], 1.0, axis=2)
        return np.einsum('uv,bvc->buc', W, O, optimize=True)

    def descend(col, F):
        for _ in range(50):
            changed = False
            for u in rng.permutation(aux):
                Fu = F[:, u, :]
                new = Fu.argmin(1)
                old = col[:, u]
                better = Fu[np.arange(len(old)), new] < Fu[np.arange(len(old)), old] - 1e-12
                idx = np.nonzero(better)[0]
                if len(idx):
                    changed = True
                    F[idx, :, old[idx]] -= W[u][None, :]
                    F[idx, :, new[idx]] += W[u][None, :]
                    col[idx, u] = new[idx]
            if not changed:
                break
        same = np.take_along_axis(F, col[:, :, None], axis=2)[:, :, 0].sum(1) / 2
        return totW - same

    F = field(col)
    val = descend(col, F)
    best_val, best_col = val.copy(), col.copy()
    for _ in range(rounds - 1):   # iterated local search: perturb a few aux vertices
        col = best_col.copy()
        m = max(1, len(aux) // 4)
        for b in range(1):
            pick = rng.integers(0, len(aux), size=(col.shape[0], m))
            col[np.arange(col.shape[0])[:, None], aux[pick]] = rng.integers(0, k, size=pick.shape)
        F = field(col)
        val = descend(col, F)
        imp = val > best_val
        best_val[imp] = val[imp]
        best_col[imp] = col[imp]
    best_val = best_val.reshape(B, R)
    j = best_val.argmax(1)
    cols = best_col.reshape(B, R, n)[np.arange(B), j]
    return best_val[np.arange(B), j], cols


# ---------------------------------------------------------------------------
# exact separation via CP-SAT (integer weights)
