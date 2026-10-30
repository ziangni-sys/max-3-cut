"""Joint refined-theorem LP over searched columns, exact CP-SAT separation, rounding.
Usage: polish_joint.py k out.json ckpt_0 ... ckpt_{k-1} [--workers N] [--scale D]"""
import sys, os, time, json, pickle, subprocess, tempfile
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, _ROOT)
import numpy as np


def exact_violations(W, k, fixed, bounds, workers, margin=1e-6):
    """For each item: a coloring with value > bound (found by the strong CP-SAT decision
    model on integer-scaled weights), or None if proved impossible (up to scaling)."""
    from gadgets.localsearch import batched_local_search
    m = W.max()
    Wi = np.rint(W / m * 1e7).astype(np.int64)
    _, hints = batched_local_search(W, k, np.asarray(fixed), 3 + k, restarts=16, rounds=4)
    thr = np.floor((np.asarray(bounds) + margin) / m * 1e7 + 0.5 * (Wi > 0).sum()).astype(np.int64)
    with tempfile.TemporaryDirectory() as td:
        np.savez(f'{td}/in.npz', W=Wi, k=k, fixed=np.asarray(fixed), hints=hints, thr=thr)
        subprocess.run([sys.executable, _ROOT + '/scripts/cpsat_worker.py',
                        f'{td}/in.npz', f'{td}/out.npz', str(workers)], check=True)
        r = np.load(f'{td}/out.npz')
    return [c if f else None for f, c in zip(r['found'], r['cols'])]


def main():
    from gadgets.jointlp import JointSymLP
    args = sys.argv[1:]
    workers = int(args[args.index('--workers') + 1]) if '--workers' in args else 3
    scale = int(args[args.index('--scale') + 1]) if '--scale' in args else 100000
    pos = [a for j, a in enumerate(args) if not a.startswith('--') and (j == 0 or not args[j - 1].startswith('--'))]
    k, out, cks = int(pos[0]), pos[1], pos[2:2 + int(pos[0])]
    cols = []
    for c in cks:
        st = pickle.load(open(c, 'rb'))
        cols.append(st['best']['cols'])
        print(f'{c}: LP alpha={1/st["best"]["g"]:.4f} cols={len(st["best"]["cols"])}', flush=True)
    J = JointSymLP(k, cols, refined=('--original' not in args), seed=0)
    tau = J.optimize()
    print(f'joint refined LP (heuristic separation): tau={tau:.7f}', flush=True)
    for rnd in range(0 if '--no-exact' in args else 4):
        added = 0
        for b in J.blocks:
            W = b.W()
            res = exact_violations(W, k, b.fixed, J.sol[b.svar], workers)
            for j, c in enumerate(res):
                if c is not None and b.add_sound_row(c, b.is_sat[j], b.items[j][1]):
                    added += 1
        tau = J.optimize()
        print(f'  exact round {rnd}: added={added} tau={tau:.7f}', flush=True)
        if added == 0:
            break
    Ws = [b.W() for b in J.blocks]
    m = max(W.max() for W in Ws)
    fam = []
    for b, W in zip(J.blocks, Ws):
        Wi = np.rint(W / m * scale).astype(np.int64)
        keep = [u for u in range(Wi.shape[0]) if u < 3 + k or Wi[u].sum() > 0]
        Wi = Wi[np.ix_(keep, keep)]
        n = Wi.shape[0]
        pos = {u: j for j, u in enumerate(keep)}
        from gadgets.symlp import PERMS
        autos = []
        for pi in range(len(PERMS)):
            autos.append([pos[b.vperm(pi, u)] + 1 for u in keep])
        fam.append({'i': b.i, 'n_vertices': n, 'automorphisms': autos,
                    'columns': [list(b.vcols[u]) for u in keep],
                    'edges': [(u + 1, v + 1, int(Wi[u, v])) for u in range(n) for v in range(u + 1, n) if Wi[u, v]]})
    json.dump({'k': k, 'lp_tau': float(tau), 'gadgets': fam}, open(out, 'w'))
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
