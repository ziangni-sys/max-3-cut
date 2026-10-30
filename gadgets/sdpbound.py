"""Rigorous Frieze-Jerrum-type SDP upper bound on  max over auxiliary colorings  of the cut
value, with the fixed vertices represented by k simplex anchors a_c.

Primal (relaxation):  max <C, X> + c0  s.t.  X >= 0 (PSD), diag(X) = 1, X[anchors] = G,
                      X_ij >= -1/(k-1) for i auxiliary (all j).
For ANY multipliers y (free), M (free, k x k), Lam >= 0 (m x N):
    <C,X> = <S,X> + sum(y) + <M,G> - <Lam~, X>   with  S = C - Diag(y) - M~ + Lam~,
and <S,X> <= N * max(0, lambda_max(S)) (trace X = N),  -<Lam~,X> <= sum(Lam)/(k-1).
Hence  UB = c0 + sum(y) + <M,G> + sum(Lam)/(k-1) + N*max(0, lambda_max(S))  is a valid bound
whatever the multipliers are (they come from an interior-point solve; numerical error can only
make UB weaker).  No highspy/ortools import here."""
import numpy as np
import cvxpy as cp


def _setup(W, k, fixed_colors, n_fixed):
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    aux = list(range(n_fixed, n))
    m = len(aux)
    const = sum(W[u, v] for u in range(n_fixed) for v in range(u + 1, n_fixed)
                if fixed_colors[u] != fixed_colors[v])
    Wc = np.zeros((m, k))
    for a, u in enumerate(aux):
        for f in range(n_fixed):
            Wc[a, int(fixed_colors[f])] += W[u, f]
    Wa = W[np.ix_(aux, aux)]
    N = k + m
    f = (k - 1) / k
    C = np.zeros((N, N))
    C[k:, k:] = -f * Wa / 2.0
    C[k:, :k] = -f * Wc / 2.0
    C[:k, k:] = -f * Wc.T / 2.0
    c0 = f * (np.triu(Wa, 1).sum() + Wc.sum()) + const
    G = (k / (k - 1)) * np.eye(k) - 1.0 / (k - 1)
    return C, c0, G, N, m


def rigorous_ub(C, c0, G, N, m, k, y, M, Lam):
    Lam = np.maximum(Lam, 0.0)
    P = np.zeros((N, N))
    P[k:, :] = Lam
    Lt = (P + P.T) / 2.0
    Mt = np.zeros((N, N))
    Mt[:k, :k] = (M + M.T) / 2.0
    S = C - np.diag(y) - Mt + Lt
    lmax = float(np.linalg.eigvalsh(S)[-1])
    return c0 + float(np.sum(y)) + float(np.sum(Mt[:k, :k] * G)) + float(Lam.sum()) / (k - 1) \
        + N * max(0.0, lmax)


def sdp_upper_bound(W, k, fixed_colors, n_fixed):
    C, c0, G, N, m = _setup(W, k, fixed_colors, n_fixed)
    X = cp.Variable((N, N), symmetric=True)
    cons = [cp.diag(X) == 1, X[:k, :k] == G, X[k:, :] >= -1.0 / (k - 1), X >> 0]
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(C, X)) + c0), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return float('inf'), float('inf'), 'error'
    y = np.asarray(cons[0].dual_value, dtype=float).ravel()
    M = np.asarray(cons[1].dual_value, dtype=float).reshape(k, k)
    Lam = np.asarray(cons[2].dual_value, dtype=float).reshape(m, N)
    best = float('inf')
    for sy in (1, -1):              # every choice gives a valid bound; keep the best
        for sm in (1, -1):
            best = min(best, rigorous_ub(C, c0, G, N, m, k, sy * y, sm * M, Lam))
    pv = float(prob.value) if prob.value is not None else float('inf')
    return best, pv, prob.status
