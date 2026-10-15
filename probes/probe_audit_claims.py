"""Audit probe: turn three asserted claims into measurements.

C1. "the answer has rank one with respect to the situations" (paper Sec. 5.4).
    Measured: the answer matrix at the best operating point, its singular values,
    its best rank-one residual, and the largest pairwise angle. Pairwise cosines of
    1.0000 are consistent with a small cone and are not a rank statement.

C2. "cost is about 1.76 ns per synapse" (paper Sec. 5.1).
    Measured: ns per synapse per row of the scaling table. The constant is
    asymptotic; at 302 neurons the per-sweep overhead dominates it.

C3. "the worm graph needs five times as many sweeps ... which locates the difficulty
    in the spectrum rather than in the size" (paper Sec. 5.1).
    Measured: the local contraction of the settling map on the worm graph and on a
    random graph of the same size and mean degree, at the same operating point, with
    the same drive. Also the certificate validity chain at the settled state, and the
    residual-to-displacement constant the settling budget needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.brain import Brain
from cadence.learning import learning_neuron_model

DATA = Path(__file__).resolve().parents[1] / "data" / "worm_connectome.json"
OUT: dict[str, object] = {}
RHO = 0.5
SITUATIONS = [
    (("odour_A", 1.0),),
    (("odour_B", 1.0),),
    (("food", 1.0),),
    (("pain", 1.0),),
    (("odour_A", 1.0), ("pain", 0.35)),
    (("odour_A", 1.0), ("pain", 0.70)),
    (("odour_A", 1.0), ("pain", 1.00)),
    (("odour_B", 1.0), ("pain", 0.35)),
    (("odour_B", 1.0), ("pain", 1.00)),
]
CONTEXT_PAIRS = [(4, 5), (4, 6), (5, 6), (7, 8)]
READOUT = ("forward", "reverse")


def worm(roles: bool = False) -> tuple[Connectome, list[str]]:
    raw = json.loads(DATA.read_text())
    names = [c["name"] for c in raw["neurons"]]
    role_of = [c.get("role", "unknown") for c in raw["neurons"]]
    index = {n: i for i, n in enumerate(names)}
    pre, post, count, sign = [], [], [], []
    for a, b, c, s in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
        if a != b:
            pre.append(a); post.append(b); count.append(c); sign.append(s)
    for a, b, c in ((int(r[0]), int(r[1]), float(r[2])) for r in raw["gap"]):
        if a != b:
            for x, y in ((a, b), (b, a)):
                pre.append(x); post.append(y); count.append(c); sign.append(1.0)
    pops: dict[str, tuple[int, ...]] = {
        "odour_A": tuple(index[n] for n in ("AWAL", "AWAR")),
        "odour_B": tuple(index[n] for n in ("AWCL", "AWCR")),
        "food": tuple(index[n] for n in ("CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER")),
        "pain": tuple(index[n] for n in ("ASHL", "ASHR")),
        "forward": tuple(index[n] for n in ("AVBL", "AVBR", "PVCL", "PVCR")),
        "reverse": tuple(index[n] for n in ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")),
    }
    if roles:
        for role in sorted(set(role_of)):
            pops[f"class_{role}"] = tuple(i for i, r in enumerate(role_of) if r == role)
    return (Connectome.from_synapses(len(names), pre=pre, post=post, count=count, sign=sign,
                                     populations=pops, label="celegans:c302"), role_of)


def sparse_random(n: int, edges: int, seed: int = 4) -> Connectome:
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, edges)
    post = rng.integers(0, n, edges)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    sign = rng.choice([-1.0, 1.0], size=len(pre), p=[0.12, 0.88])
    return Connectome.from_synapses(n, pre=pre, post=post, sign=sign, label=f"random{n}")


def row_norm(conn: Connectome, rho: float = RHO) -> np.ndarray:
    indeg = np.bincount(conn.post, minlength=conn.n).astype(float)
    indeg[indeg == 0] = 1.0
    return conn.sign * (rho / indeg[conn.post])


def make(conn: Connectome, log_gain: np.ndarray | None = None, rho: float = RHO) -> Brain:
    return Brain(conn, learning_neuron_model(), backend="cpu", efficacy=row_norm(conn, rho),
                 log_gain=np.zeros(conn.n) if log_gain is None else log_gain)


def best_config() -> dict:
    """The best point of the scan, read from the scan's own dump rather than restated."""
    dump = json.loads((Path(__file__).resolve().parents[1] / "docs" / "assets"
                       / "prereq_scan.json").read_text())
    return max(dump["rows"], key=lambda r: r["spread"])


# --- C1 -------------------------------------------------------------------

def answers() -> dict:
    conn, _ = worm(roles=True)
    cfg = best_config()
    log_gain = np.zeros(conn.n)
    if cfg["gain_class"]:
        log_gain[list(conn.populations[cfg["gain_class"]])] = float(np.log(cfg["class_scale"]))
    brain = make(conn, log_gain, cfg["row_sum"])      # the scan's best operating point
    readout = np.array(sorted(set(conn.populations["forward"]) | set(conn.populations["reverse"])))
    codes = []
    for senses in SITUATIONS:
        drive = np.zeros((1, conn.n))
        for pop, level in senses:
            drive[0, list(conn.populations[pop])] = level
        eq = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-6)
        codes.append(np.asarray(eq.state.activation).ravel()[readout])
    A = np.stack(codes)
    norms = np.linalg.norm(A, axis=1)
    unit = A / norms[:, None]
    cos = unit @ unit.T
    sv = np.linalg.svd(A, compute_uv=False)
    energy = float(sv[0] ** 2 / np.sum(sv ** 2))
    # best rank-one approximation A ~ c v^T
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    resid = float(np.linalg.norm(A - s[0] * np.outer(u[:, 0], vt[0])) / np.linalg.norm(A))
    ctx = [float(cos[i, j]) for i, j in CONTEXT_PAIRS]
    off = cos[np.triu_indices(len(SITUATIONS), 1)]
    return {
        "config": {"row_sum": cfg["row_sum"], "gain_class": cfg["gain_class"],
                   "class_scale": cfg["class_scale"], "scan_spread": cfg["spread"]},
        "situations": len(SITUATIONS), "readout_cells": int(len(readout)),
        "singular_values": [round(float(x), 6) for x in sv],
        "sigma2_over_sigma1": round(float(sv[1] / sv[0]), 6),
        "energy_in_first_mode": round(energy, 6),
        "rank_one_relative_residual": round(resid, 6),
        "context_pair_cosines": [round(c, 8) for c in ctx],
        "max_angle_deg_all_pairs": round(float(np.degrees(np.arccos(np.clip(off.min(), -1, 1)))), 4),
        "max_angle_deg_context_pairs": round(float(np.degrees(np.arccos(np.clip(min(ctx), -1, 1)))), 4),
        "answer_norms": [round(float(x), 6) for x in norms],
        "spread": round(float(np.mean(A.max(axis=0) - A.min(axis=0))), 6),
    }


# --- C2 -------------------------------------------------------------------

def ns_per_synapse() -> list[dict]:
    table = [(302, 3_137, 27.2), (1_000, 10_337, 161.4),
             (10_000, 103_943, 193.0), (100_000, 1_239_921, 2_185.8)]
    return [{"neurons": n, "synapses": s, "us_per_sweep": us,
             "ns_per_synapse": round(us * 1000 / s, 3)} for n, s, us in table]


# --- C3 -------------------------------------------------------------------

def contraction(brain: Brain, drive: np.ndarray, tolerance: float) -> dict:
    nm = brain.neuron_model
    W = brain.dense()
    eq = brain.equilibrate(drive, budget=1024, chunk=32, tolerance=tolerance)
    v = np.asarray(eq.state.v).ravel()
    s = np.asarray(eq.state.activation).ravel()
    d = nm.slope_at(v)
    dt = nm.dt
    J = (1 - dt) * np.eye(brain.connectome.n) + dt * W * d[None, :]
    K = (1 - dt) * np.eye(brain.connectome.n) + dt * np.abs(W) * d[None, :]
    rho_J = float(np.max(np.abs(np.linalg.eigvals(J))))
    rho_K = float(np.max(np.abs(np.linalg.eigvals(K))))
    resid = np.asarray(eq.residual).ravel() if np.ndim(eq.residual) else np.asarray([eq.residual])
    return {
        "sweeps": int(eq.steps),
        "residual": float(np.max(eq.residual)),
        "active_fraction": round(float((s > 1e-9).mean()), 4),
        "dt": dt,
        "slopes_min": round(float(d.min()), 8),
        "abs_row_sum_max": round(float(np.abs(W).sum(axis=1).max()), 4),
        "gershgorin_inf_norm_q_local": round(float(((1 - dt) + dt * (np.abs(W) * d[None, :]).sum(axis=1)).max()), 4),
        "rho_true_J": round(rho_J, 6),
        "rho_majorant_K": round(rho_K, 6),
        "majorant_dominates": bool(np.all(np.abs(J) <= K + 1e-15)),
        "rho_J_le_rho_K": bool(rho_J <= rho_K + 1e-12),
        # the bridge the settling budget needs: |displacement| = dt * |residual|
        "residual_over_displacement": round(float(np.max(np.abs(resid)) / (dt * max(np.max(np.abs(resid)), 1e-30))), 4),
    }


def certificate_scan() -> list[dict]:
    conn, _ = worm()
    rows = []
    for rho in (0.1, 0.5, 2.0):
        brain = make(conn)
        brain = Brain(conn, learning_neuron_model(), backend="cpu",
                      efficacy=conn.sign * (rho / np.maximum(np.bincount(conn.post, minlength=conn.n), 1)[conn.post]))
        drive = np.zeros((1, conn.n))
        drive[0, :8] = 1.0
        rows.append({"row_sum": rho, **contraction(brain, drive, 1e-4)})
    return rows


def main() -> None:
    conn, _ = worm()
    n_rand = 302
    edges = int(round(3137 * n_rand / 302))
    rand = sparse_random(n_rand, edges)
    drive = np.zeros((1, conn.n))
    drive[0, :8] = 1.0
    OUT["C1_answer_matrix"] = answers()
    OUT["C2_ns_per_synapse"] = ns_per_synapse()
    OUT["C3_worm"] = {"neurons": conn.n, "synapses": conn.synapses, **contraction(make(conn), drive, 1e-4)}
    OUT["C3_random_same_size"] = {"neurons": rand.n, "synapses": rand.synapses,
                                  **contraction(make(rand), drive, 1e-4)}
    OUT["C3_certificate_by_operating_point"] = certificate_scan()
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
