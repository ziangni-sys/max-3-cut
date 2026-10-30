"""Exact max over auxiliary colorings via CP-SAT (kept free of highspy imports:
OR-tools and highspy ship clashing HiGHS symbols)."""
import numpy as np


def cpsat_max(Wint, k, fixed_colors, n_fixed, time_limit=60.0, workers=1, hint=None):
    from ortools.sat.python import cp_model
    n = Wint.shape[0]
    aux = range(n_fixed, n)
    m = cp_model.CpModel()
    b = {(u, c): m.NewBoolVar(f'b{u}_{c}') for u in aux for c in range(k)}
    for u in aux:
        m.AddExactlyOne([b[u, c] for c in range(k)])
    obj = []
    const = 0
    for u in range(n_fixed):
        for v in range(u + 1, n_fixed):
            if Wint[u, v] and fixed_colors[u] != fixed_colors[v]:
                const += int(Wint[u, v])
    for u in aux:
        for c in range(k):
            g = sum(int(Wint[u, v]) for v in range(n_fixed) if fixed_colors[v] != c)
            if g:
                obj.append(g * b[u, c])
    for u in aux:
        for v in aux:
            if v <= u or not Wint[u, v]:
                continue
            e = m.NewBoolVar(f'e{u}_{v}')
            for c in range(k):
                m.AddBoolOr([e.Not(), b[u, c].Not(), b[v, c].Not()])
            obj.append(int(Wint[u, v]) * e)
    if hint is not None:
        for u in aux:
            for c in range(k):
                m.AddHint(b[u, c], int(hint[u]) == c)
    m.Maximize(sum(obj))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    st = s.Solve(m)
    if st != cp_model.OPTIMAL:
        raise RuntimeError(f'CP-SAT status {s.StatusName(st)}')
    col = list(fixed_colors) + [next(c for c in range(k) if s.Value(b[u, c])) for u in aux]
    return int(round(s.ObjectiveValue())) + const, col


def cpsat_decide(Wint, k, fixed_colors, n_fixed, threshold, time_limit=3600.0, workers=1, hint=None):
    """Decide whether some auxiliary coloring has cut value > threshold (integer weights).
    Returns None if CP-SAT proves max <= threshold, else (value, coloring) of a witness
    with value > threshold.  Raises if undecided within the time limit."""
    from ortools.sat.python import cp_model
    n = Wint.shape[0]
    aux = range(n_fixed, n)
    m = cp_model.CpModel()
    b = {(u, c): m.NewBoolVar(f'b{u}_{c}') for u in aux for c in range(k)}
    for u in aux:
        m.AddExactlyOne([b[u, c] for c in range(k)])
    obj = []
    const = 0
    for u in range(n_fixed):
        for v in range(u + 1, n_fixed):
            if Wint[u, v] and fixed_colors[u] != fixed_colors[v]:
                const += int(Wint[u, v])
    for u in aux:
        for c in range(k):
            g = sum(int(Wint[u, v]) for v in range(n_fixed) if fixed_colors[v] != c)
            if g:
                obj.append(g * b[u, c])
    for u in aux:
        for v in aux:
            if v <= u or not Wint[u, v]:
                continue
            e = m.NewBoolVar(f'e{u}_{v}')
            for c in range(k):
                m.AddBoolOr([e.Not(), b[u, c].Not(), b[v, c].Not()])
            obj.append(int(Wint[u, v]) * e)
    m.Add(sum(obj) >= int(threshold) - const + 1)
    if hint is not None:
        for u in aux:
            for c in range(k):
                m.AddHint(b[u, c], int(hint[u]) == c)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    st = s.Solve(m)
    if st == cp_model.INFEASIBLE:
        return None
    if st in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        col = list(fixed_colors) + [next(c for c in range(k) if s.Value(b[u, c])) for u in aux]
        colv = np.array(col)
        val = int((Wint * (colv[:, None] != colv[None, :])).sum() // 2)
        return val, col
    raise RuntimeError(f'CP-SAT undecided: {s.StatusName(st)}')


def cpsat_decide_strong(Wint, k, fixed_colors, n_fixed, threshold, time_limit=3600.0, workers=1, hint=None, value_precedence=False):
    """Same as cpsat_decide, with a 'same colour' variable for every pair of auxiliary
    vertices plus transitivity constraints (tighter LP relaxation)."""
    from ortools.sat.python import cp_model
    from itertools import combinations
    n = Wint.shape[0]
    aux = list(range(n_fixed, n))
    m = cp_model.CpModel()
    b = {(u, c): m.NewBoolVar(f'b{u}_{c}') for u in aux for c in range(k)}
    for u in aux:
        m.AddExactlyOne([b[u, c] for c in range(k)])
    s = {}
    for u, v in combinations(aux, 2):
        s[u, v] = m.NewBoolVar(f's{u}_{v}')
        for c in range(k):              # same colour c => s = 1 ; s = 1 => colours equal
            m.AddBoolOr([b[u, c].Not(), b[v, c].Not(), s[u, v]])
            m.AddBoolOr([s[u, v].Not(), b[u, c].Not(), b[v, c]])
    for u, v, w in combinations(aux, 3):  # transitivity
        m.Add(s[u, v] + s[v, w] - s[u, w] <= 1)
        m.Add(s[u, v] - s[v, w] + s[u, w] <= 1)
        m.Add(-s[u, v] + s[v, w] + s[u, w] <= 1)
    const = 0
    for u in range(n_fixed):
        for v in range(u + 1, n_fixed):
            if Wint[u, v] and fixed_colors[u] != fixed_colors[v]:
                const += int(Wint[u, v])
    obj = []
    for u in aux:
        for c in range(k):
            g = sum(int(Wint[u, v]) for v in range(n_fixed) if fixed_colors[v] != c)
            if g:
                obj.append(g * b[u, c])
    for (u, v), sv in s.items():
        if Wint[u, v]:
            obj.append(int(Wint[u, v]) * (1 - sv))
    m.Add(sum(obj) >= int(threshold) - const + 1)
    if value_precedence:
        # colours used by no fixed vertex are interchangeable: require the first auxiliary
        # vertex using free colour f_{j+1} to come after the first one using f_j
        # (value precedence over the auxiliary order), which keeps one coloring per class
        free = [c for c in range(k) if c not in set(int(v) for v in fixed_colors)]
        for a, bcol in zip(free, free[1:]):
            # for every aux vertex u: if u has colour bcol then some earlier-or-equal aux vertex has colour a
            for t, u in enumerate(aux):
                prev = [b[w, a] for w in aux[:t]]
                m.AddBoolOr(prev + [b[u, bcol].Not()])
        hint = None   # the hint may violate the precedence; drop it
    if hint is not None:
        for u in aux:
            for c in range(k):
                m.AddHint(b[u, c], int(hint[u]) == c)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit
    sol.parameters.num_search_workers = workers
    sol.parameters.linearization_level = 2
    st = sol.Solve(m)
    if st == cp_model.INFEASIBLE:
        return None
    if st in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        col = list(fixed_colors) + [next(c for c in range(k) if sol.Value(b[u, c])) for u in aux]
        colv = np.array(col)
        return int((Wint * (colv[:, None] != colv[None, :])).sum() // 2), col
    raise RuntimeError(f'CP-SAT undecided: {sol.StatusName(st)}')
