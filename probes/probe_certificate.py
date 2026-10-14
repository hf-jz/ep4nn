"""Probe 5: worm-scale settle cost and a state-dependent contraction certificate.

A. The library's transport choice at 302 neurons (blocked dense vs CSR), end to end.
B. The paper's global worst-case bound versus the actual local Jacobian on the
   active set: max_i (1-dt) + dt * sum_j |W_ij| * s'(v_j) < 1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.brain import Brain
from cadence.learning import learning_neuron_model

OUT: dict[str, object] = {}


def build() -> Connectome:
    raw = json.loads(Path("/Users/sheng/math/ep4nn/data/worm_connectome.json").read_text())
    pre, post, count, sign = [], [], [], []
    for a, b, c, s in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
        if a != b:
            pre.append(a); post.append(b); count.append(c); sign.append(s)
    for a, b, c in ((int(r[0]), int(r[1]), float(r[2])) for r in raw["gap"]):
        if a != b:
            for x, y in ((a, b), (b, a)):
                pre.append(x); post.append(y); count.append(c); sign.append(1.0)
    return Connectome.from_synapses(len(raw["neurons"]), pre=pre, post=post, count=count,
                                    sign=sign, label="celegans:c302_A_Full")


def settle_cost(conn: Connectome, dense_limit: int, reps: int = 20) -> dict:
    brain = Brain(conn, learning_neuron_model(), backend="cpu", dense_limit=dense_limit)
    drive = np.zeros((1, conn.n))
    drive[0, :4] = 1.0
    sweeps, times = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        eq = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4)
        times.append(time.perf_counter() - t0)
        sweeps.append(int(eq.steps))
    # single sweep, cold, same path the settle uses
    t0 = time.perf_counter()
    for _ in range(300):
        brain.settle_batch(drive, steps=1, state=None)
    return {
        "dense_limit": dense_limit,
        "blocked": bool(brain._blocked),
        "sweeps": int(np.mean(sweeps)),
        "ms_per_settle": round(1000 * float(np.mean(times)), 3),
        "sweep_us": round(1e6 * (time.perf_counter() - t0) / 300, 1),
        "residual": float(np.max(eq.residual)),
    }


def certificate(conn: Connectome) -> dict:
    brain = Brain(conn, learning_neuron_model(), backend="cpu")
    nm = brain.neuron_model
    W = brain.dense()
    drive = np.zeros((1, conn.n))
    drive[0, :4] = 1.0
    eq = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4)
    v = np.asarray(eq.state.v).ravel()
    s = np.asarray(eq.state.activation).ravel()
    d = nm.slope_at(v)
    dt = nm.dt
    rows_abs = np.abs(W).sum(axis=1)
    max_d = float(d.max())
    q_global = (1 - dt) + dt * float(np.max(rows_abs)) * max_d
    weighted = np.abs(W) * d[None, :]
    row_local = weighted.sum(axis=1)
    q_local = (1 - dt) + dt * row_local
    K = (1 - dt) * np.eye(conn.n) + dt * weighted
    rad = float(np.max(np.abs(np.linalg.eigvals(K))))
    witness = None
    if rad < 1.0:
        alpha_guess = min(0.999, max(0.5 * (1 + rad), rad + 1e-6))
        h = np.ones(conn.n)
        for _ in range(500):
            h = (K / alpha_guess) @ h + 1.0
        alpha = float(np.max((K @ h) / h))
        witness = {"alpha": round(alpha, 6), "h_min": float(h.min()), "h_max": float(h.max())}
    return {
        "dt": dt,
        "activation_lipschitz_max": round(max_d, 4),
        "active_neurons": int((s > 1e-9).sum()),
        "active_fraction": round(float((s > 1e-9).mean()), 4),
        "q_global_worst_case": round(q_global, 3),
        "q_local_max_row": round(float(q_local.max()), 3),
        "rows_over_1_global": int((((1 - dt) + dt * rows_abs * max_d) >= 1.0).sum()),
        "rows_over_1_local": int((q_local >= 1.0).sum()),
        "spectral_radius_local_K": round(rad, 4),
        "weighted_witness": witness,
        "settled_residual": float(np.max(eq.residual)),
    }


def main() -> None:
    conn = build()
    OUT["settle_blocked_default"] = settle_cost(conn, 2048)
    OUT["settle_csr_forced"] = settle_cost(conn, 0)
    OUT["certificate"] = certificate(conn)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
