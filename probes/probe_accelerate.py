"""Probe 2: can the per-tick settle be made cheaper? Anderson acceleration + delta sparsity.

Plain sweep: x <- F(x) with F one integration step (brain.settle_batch steps=1).
Anderson type-I mixing over the last m iterates, with a monotone safeguard.
Measured on the same composed brain the library's `live` uses.
"""

from __future__ import annotations

import json
import time

import numpy as np

from cadence import Brain
from cadence.brain import BrainState

M = 5
LAM = 1e-8


def make_brain(modules=(64, 32, 16), inputs=8, actions=4, seed=3) -> Brain:
    return Brain.compose(inputs=inputs, actions=actions, modules=modules, arousal=True, seed=seed)


def state_of(core, v: np.ndarray) -> BrainState:
    return BrainState(
        v=v,
        activation=core.neuron_model.activation(v),
        adaptation=np.zeros_like(v),
        steps=0,
    )


def sweep(core, drive, v):
    st = core.settle_batch(drive, steps=1, state=state_of(core, v))
    return st.v


def residual(core, drive, v) -> float:
    return float(np.max(core.residual(drive, state_of(core, v))))


def anderson(core, drive, v0, budget=256, tol=1e-6, m=M):
    """Anderson mixing on x <- F(x), one F evaluation per iteration, monotone safeguard."""
    shape = v0.shape
    v = v0.copy()
    f = sweep(core, drive, v)
    rs: list[np.ndarray] = [(f - v).ravel().copy()]
    fs: list[np.ndarray] = [f.ravel().copy()]
    it = 0
    while it < budget:
        it += 1
        k = min(m, len(rs) - 1)
        cand = None
        if k >= 1:
            base = rs[-1]
            dif = np.stack([rs[-1 - j] - base for j in range(1, k + 1)], axis=1)
            fbase = fs[-1]
            fdif = np.stack([fs[-1 - j] - fbase for j in range(1, k + 1)], axis=1)
            try:
                beta = np.linalg.solve(dif.T @ dif + LAM * np.eye(k), -dif.T @ base)
                cand = (fbase + fdif @ beta).reshape(shape)
            except np.linalg.LinAlgError:
                cand = None
        if cand is None or not np.isfinite(cand).all():
            cand = f
        f_cand = sweep(core, drive, cand)
        accepted = np.linalg.norm(f_cand - cand) <= np.linalg.norm(f - v)
        if accepted:
            v, f = cand, f_cand
        else:  # fall back to the plain step, never worse than the base iteration
            v, f = f, sweep(core, drive, f)
        rs.append((f - v).ravel().copy())
        fs.append(f.ravel().copy())
        if len(rs) > m + 2:
            rs.pop(0)
            fs.pop(0)
        if residual(core, drive, v) <= tol:
            break
    return v, it


def plain(core, drive, v0, budget=2048, tol=1e-6):
    v = v0.copy()
    it = 0
    while it < budget:
        it += 1
        v = sweep(core, drive, v)
        if residual(core, drive, v) <= tol:
            break
    return v, it


def delta_sparsity(core, drive, v0, steps=40):
    v = v0.copy()
    s = core.neuron_model.activation(v)
    rows = []
    for _ in range(steps):
        prev = s
        st = core.settle_batch(drive, steps=1, state=state_of(core, v))
        v = st.v
        s = st.activation
        d = np.abs(np.asarray(s) - np.asarray(prev))
        rows.append(
            {
                "max": float(d.max()),
                "frac_gt_1e-3": float((d > 1e-3).mean()),
                "frac_gt_1e-5": float((d > 1e-5).mean()),
            }
        )
    return rows


def transport_cost(core, reps=300):
    n, syn = core.connectome.n, core.connectome.synapses
    pre, post = core.connectome.pre, core.connectome.post
    starts, targets = core._starts, core._neurons_with_synaptic_input
    w = core.weights
    s = np.full((1, n), 0.1)
    t0 = time.perf_counter()
    for _ in range(reps):
        msg = s[:, pre] * w
        out = np.zeros((1, n))
        out[:, targets] = np.add.reduceat(msg, starts, axis=1)
    t_reduceat = (time.perf_counter() - t0) / reps
    from cadence.sparse import transport

    M = transport(core.connectome, w)
    t_csr = None
    if M is not None:
        t0 = time.perf_counter()
        for _ in range(reps):
            M.dot(s.T).T
        t_csr = (time.perf_counter() - t0) / reps
    out_lines = {}
    for frac in (0.05, 0.2, 1.0):
        k = max(1, int(round(n * frac)))
        sel = np.arange(k)
        d = np.zeros((1, n))
        d[0, sel] = 1.0
        mask = np.isin(pre, sel)
        parts = post[mask]
        pw = w[mask]
        t0 = time.perf_counter()
        for _ in range(reps):
            out2 = np.zeros(n)
            np.add.at(out2, parts, d[0, pre[mask]] * pw)
        out_lines[f"delta_scatter_frac_{frac}"] = {
            "us": round(1e6 * (time.perf_counter() - t0) / reps, 1),
            "edges": int(mask.sum()),
        }
    return {
        "synapses": int(syn),
        "neurons": int(n),
        "segmented_reduceat_us": round(1e6 * t_reduceat, 1),
        "csr_dot_us": None if t_csr is None else round(1e6 * t_csr, 1),
        **out_lines,
    }


def main() -> None:
    out = {}
    for modules in [(64, 32, 16), (256, 128, 64)]:
        b = make_brain(modules=modules)
        core = b.brain
        drive = np.zeros((1, core.connectome.n))
        drive[0, list(b.sensory_index)] = 1.0
        v0 = np.zeros((1, core.connectome.n))
        # cold-start comparison at three tolerances
        for tol in (1e-3, 1e-5, 1e-6):
            t0 = time.perf_counter()
            _, it_p = plain(core, drive, v0, budget=4096, tol=tol)
            t_p = time.perf_counter() - t0
            t0 = time.perf_counter()
            v_a, it_a = anderson(core, drive, v0, budget=512, tol=tol)
            t_a = time.perf_counter() - t0
            out.setdefault(f"{modules}", {})[f"tol={tol:g}"] = {
                "plain_iters": it_p,
                "plain_ms": round(1000 * t_p, 3),
                "anderson_iters": it_a,
                "anderson_ms": round(1000 * t_a, 3),
                "anderson_residual": residual(core, drive, v_a),
                "speedup_iters": round(it_p / max(it_a, 1), 2),
                "speedup_time": round(t_p / max(t_a, 1e-9), 2),
            }
        # warm start with a small input change (the control case)
        drive2 = drive.copy()
        drive2[0, list(b.sensory_index)[0]] = 0.9
        v_warm = core.settle_batch(drive, steps=32, state=None).v
        for tol in (1e-5,):
            _, it_p = plain(core, drive2, v_warm, budget=4096, tol=tol)
            _, it_a = anderson(core, drive2, v_warm, budget=512, tol=tol)
            out.setdefault(f"{modules}", {})[f"warm_delta_tol={tol:g}"] = {
                "plain_iters": it_p,
                "anderson_iters": it_a,
            }
        out.setdefault(f"{modules}", {})["delta_sparsity"] = delta_sparsity(core, drive, v0, 40)
        out.setdefault(f"{modules}", {})["transport_cost"] = transport_cost(core)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
