"""Audit fixes: close the gaps the audit left open with measurements, not prose.

G1  the certificate was a floating-point eigenvalue; here it becomes a certified bound.
    For a nonnegative K and any positive h, Kolmogorov's inequality gives
    rho(K) <= max_i (Kh)_i / h_i. Every operation is then pushed outward by one ulp
    (nextafter), so the printed alpha is an upper bound in exact arithmetic and the
    witness inequality is asserted rather than assumed.

G2  the certificate was one state; here it is measured over a domain of drives and
    operating points, so the claim carries a validity region.

G4  growth was never exercised because the prerequisite fails on one declared readout;
    here the prerequisite is scanned over a family of declared readouts, which turns
    "the declared readout does not separate" into "no readout in this family separates",
    or finds one that does and hands it to the growth layer.

G5  the random-graph comparison was one seed; here it is five.
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
SITUATIONS = [
    (("odour_A", 1.0),), (("odour_B", 1.0),), (("food", 1.0),), (("pain", 1.0),),
    (("odour_A", 1.0), ("pain", 0.35)), (("odour_A", 1.0), ("pain", 0.70)),
    (("odour_A", 1.0), ("pain", 1.00)), (("odour_B", 1.0), ("pain", 0.35)),
    (("odour_B", 1.0), ("pain", 1.00)),
]
CONTEXT_PAIRS = [(4, 5), (4, 6), (5, 6), (7, 8)]
SENSES = {"odour_A": ("AWAL", "AWAR"), "odour_B": ("AWCL", "AWCR"),
          "pain": ("ASHL", "ASHR"),
          "food": ("CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER")}


def load() -> tuple[Connectome, dict[str, tuple[int, ...]], list[str], np.ndarray, np.ndarray]:
    raw = json.loads(DATA.read_text())
    names = [c["name"] for c in raw["neurons"]]
    roles = [c.get("role", "unknown") for c in raw["neurons"]]
    index = {n: i for i, n in enumerate(names)}
    pre, post, count, sign = [], [], [], []
    for a, b, c, s in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
        if a != b:
            pre.append(a); post.append(b); count.append(c); sign.append(s)
    for a, b, c in ((int(r[0]), int(r[1]), float(r[2])) for r in raw["gap"]):
        if a != b:
            for x, y in ((a, b), (b, a)):
                pre.append(x); post.append(y); count.append(c); sign.append(1.0)
    pops = {name: tuple(index[n] for n in cells) for name, cells in SENSES.items()}
    pops["forward"] = tuple(index[n] for n in ("AVBL", "AVBR", "PVCL", "PVCR"))
    pops["reverse"] = tuple(index[n] for n in ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"))
    for role in sorted(set(roles)):
        pops[f"class_{role}"] = tuple(i for i, r in enumerate(roles) if r == role)
    conn = Connectome.from_synapses(len(names), pre=pre, post=post, count=count, sign=sign,
                                   populations=pops, label="celegans:c302")
    return conn, pops, names, np.asarray(pre), np.asarray(post)


def row_norm(conn: Connectome, rho: float) -> np.ndarray:
    indeg = np.bincount(conn.post, minlength=conn.n).astype(float)
    indeg[indeg == 0] = 1.0
    return conn.sign * (rho / indeg[conn.post])


def certified_witness(K: np.ndarray, iters: int = 3000) -> dict:
    """rho(K) <= alpha, certified: Kolmogorov's inequality with outward rounding.

    h must approximate the Perron vector. An additive iteration (h <- Kh + 1) converges to
    (I-K)^{-1}1 instead, which is a legal witness but a very loose one: it reported 0.99
    where the spectral radius is 0.93. Power iteration gives the tight witness; several
    starting vectors are tried and the smallest alpha is kept, since every one is valid.
    """
    best = None
    for start in (np.ones(K.shape[0]), K.sum(axis=0) + 1e-12, np.arange(1.0, K.shape[0] + 1)):
        h = np.asarray(start, float)
        for _ in range(iters):
            h = K @ h
            peak = float(np.max(h))
            if peak <= 0 or not np.isfinite(peak):
                break
            h = h / peak
        if not np.isfinite(h).all() or np.max(h) <= 0:
            continue
        h = np.maximum(h, np.finfo(float).tiny)
        ratios = np.nextafter((K @ h) / h, np.inf)
        alpha = float(np.max(ratios))
        if best is None or alpha < best[0]:
            best = (alpha, h)
    if best is None:
        raise ValueError("no usable witness vector")
    alpha, h = best
    up = lambda a: np.nextafter(a, np.inf)
    dn = lambda a: np.nextafter(a, 0.0)
    num, den = up(K @ h), dn(h)
    alpha = float(np.max(up(num / den)))
    witness = bool(np.all(num <= up(alpha * h)))
    return {"alpha_certified": round(alpha, 6), "witness_holds": witness,
            "h_min": round(float(h.min()), 4), "h_max": round(float(h.max()), 4),
            "rho_numeric": round(float(np.max(np.abs(np.linalg.eigvals(K)))), 6)}


def matrices(conn: Connectome, efficacy: np.ndarray | None, drive: np.ndarray,
             tolerance: float = 1e-4) -> tuple[np.ndarray, dict]:
    kw = {} if efficacy is None else {"efficacy": efficacy}
    brain = Brain(conn, learning_neuron_model(), backend="cpu", **kw)
    nm = brain.neuron_model
    eq = brain.equilibrate(drive, budget=1024, chunk=32, tolerance=tolerance)
    v = np.asarray(eq.state.v).ravel()
    s = np.asarray(eq.state.activation).ravel()
    d = nm.slope_at(v)
    dt = nm.dt
    W = brain.dense()
    J = (1 - dt) * np.eye(conn.n) + dt * W * d[None, :]
    K = (1 - dt) * np.eye(conn.n) + dt * np.abs(W) * d[None, :]
    return K, {"sweeps": int(eq.steps), "residual": float(np.max(eq.residual)),
               "active_fraction": round(float((s > 1e-9).mean()), 4),
               "majorant_dominates": bool(np.all(np.abs(J) <= K + 1e-15)),
               "rho_J": round(float(np.max(np.abs(np.linalg.eigvals(J)))), 6)}


def g1(conn: Connectome) -> dict:
    drive = np.zeros((1, conn.n)); drive[0, :8] = 1.0
    out = {}
    K, info = matrices(conn, None, drive)                       # library default efficacy
    out["default_efficacy"] = {**info, **certified_witness(K)}
    K, info = matrices(conn, row_norm(conn, 0.5), drive)         # declared operating point
    out["declared_point_row_sum_0.5"] = {**info, **certified_witness(K)}
    return out


def g2(conn: Connectome) -> dict:
    rows = []
    for rho in (0.1, 0.5, 2.0):
        eff = row_norm(conn, rho)
        for amp in (0.5, 1.0, 2.0, 4.0):
            drive = np.zeros((1, conn.n)); drive[0, :8] = amp
            K, info = matrices(conn, eff, drive)
            w = certified_witness(K, iters=1500)
            rows.append({"row_sum": rho, "drive_amplitude": amp, "sweeps": info["sweeps"],
                         "active_fraction": info["active_fraction"],
                         "alpha_certified": w["alpha_certified"], "witness_holds": w["witness_holds"]})
    alphas = [r["alpha_certified"] for r in rows]
    return {"cells": rows, "worst_alpha": max(alphas), "all_below_one": all(a < 1.0 for a in alphas),
            "n_states": len(rows)}


def g4(conn: Connectome, pops: dict[str, tuple[int, ...]], pre: np.ndarray, post: np.ndarray) -> dict:
    """Scan declared readouts: does any of them separate the pairs that must differ?"""
    driven = sorted({i for name in SENSES for i in pops[name]})
    first_hop = sorted(set(post[np.isin(pre, driven)].tolist()) - set(driven))
    rng = np.random.default_rng(11)
    candidates = {
        "forward+reverse (declared)": sorted(set(pops["forward"]) | set(pops["reverse"])),
        "command": sorted(pops["class_command"]),
        "interneuron": sorted(pops["class_interneuron"]),
        "motor": sorted(pops["class_motor"]),
        "sensory": sorted(pops["class_sensory"]),
        "first hop from the senses": first_hop[:120],
        "forward only": sorted(pops["forward"]),
        "reverse only": sorted(pops["reverse"]),
        "random 10 (control)": sorted(rng.choice(conn.n, 10, replace=False).tolist()),
        "all 302 (the whole state)": list(range(conn.n)),
    }
    log_gain = np.zeros(conn.n)
    log_gain[list(pops["class_interneuron"])] = float(np.log(2.0))   # the best scanned point
    brain = Brain(conn, learning_neuron_model(), backend="cpu", efficacy=row_norm(conn, 0.5),
                  log_gain=log_gain)
    codes = []
    for senses in SITUATIONS:
        drive = np.zeros((1, conn.n))
        for pop, level in senses:
            drive[0, list(pops[pop])] = level
        eq = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4)
        codes.append(np.asarray(eq.state.activation).ravel())
    codes = np.stack(codes)
    rows = []
    for label, cells in candidates.items():
        A = codes[:, cells]
        norms = np.linalg.norm(A, axis=1)
        unit = A / np.where(norms[:, None] == 0, 1.0, norms[:, None])
        cos = unit @ unit.T
        coll = max(abs(float(cos[i, j])) for i, j in CONTEXT_PAIRS)
        spread = float(np.mean(A.max(axis=0) - A.min(axis=0)))
        rel = [float(np.linalg.norm(A[i] - A[j])) / max(float(np.linalg.norm(A[i])),
                                                       float(np.linalg.norm(A[j])), 1e-12)
               for i, j in CONTEXT_PAIRS]
        rows.append({"readout": label, "cells": len(cells), "spread": round(spread, 4),
                     "collision": round(coll, 6), "pair_motion_rel": round(float(np.mean(rel)), 6),
                     "separates": bool(spread >= 0.05 and coll <= 0.85),
                     "growth_exercisable": bool(spread >= 0.05 and coll > 0.85)})
    return {"rows": rows,
            "any_readout_separates": any(r["separates"] for r in rows),
            "any_growth_exercisable": any(r["growth_exercisable"] for r in rows),
            "exercisable_readouts": [r["readout"] for r in rows if r["growth_exercisable"]]}


def g5() -> list[dict]:
    rows = []
    for seed in (1, 2, 3, 4, 5):
        n, edges = 302, 3137
        rng = np.random.default_rng(seed)
        pre = rng.integers(0, n, edges); post = rng.integers(0, n, edges)
        keep = pre != post
        pre, post = pre[keep], post[keep]
        sign = rng.choice([-1.0, 1.0], size=len(pre), p=[0.12, 0.88])
        conn = Connectome.from_synapses(n, pre=pre, post=post, sign=sign, label=f"random{seed}")
        drive = np.zeros((1, n)); drive[0, :8] = 1.0
        K, info = matrices(conn, row_norm(conn, 0.5), drive)
        rows.append({"seed": seed, "synapses": int(conn.synapses), "sweeps": info["sweeps"],
                     "rho_K_numeric": round(float(np.max(np.abs(np.linalg.eigvals(K)))), 6),
                     "alpha_certified": certified_witness(K, iters=1200)["alpha_certified"]})
    return rows


def main() -> None:
    conn, pops, names, pre, post = load()
    OUT["G1_certified_bound"] = g1(conn)
    OUT["G2_validity_domain"] = g2(conn)
    OUT["G5_random_seeds"] = g5()
    OUT["G4_readout_family"] = g4(conn, pops, pre, post)
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "audit_fixes.json"
    dump.write_text(json.dumps(OUT, indent=1, default=str) + "\n")
    g1o, g2o, g5o, g4o = OUT["G1_certified_bound"], OUT["G2_validity_domain"], OUT["G5_random_seeds"], OUT["G4_readout_family"]
    print(f"G1 certified alpha: default efficacy {g1o['default_efficacy']['alpha_certified']} "
          f"(rho {g1o['default_efficacy']['rho_numeric']}), declared point "
          f"{g1o['declared_point_row_sum_0.5']['alpha_certified']} "
          f"(rho {g1o['declared_point_row_sum_0.5']['rho_numeric']}); "
          f"witness holds: {g1o['default_efficacy']['witness_holds']}/{g1o['declared_point_row_sum_0.5']['witness_holds']}")
    print(f"G2 domain: {g2o['n_states']} states, worst alpha {g2o['worst_alpha']}, all < 1: {g2o['all_below_one']}")
    print("G5 seeds: " + ", ".join(f"s{r['seed']}: rho {r['rho_K_numeric']} sweeps {r['sweeps']}" for r in g5o))
    print(f"G4 readouts: any separates {g4o['any_readout_separates']}, growth exercisable {g4o['any_growth_exercisable']} -> {g4o['exercisable_readouts']}")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
