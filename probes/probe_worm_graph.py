"""Probe 4: the real 302-neuron C. elegans wiring, measured as a Cadence brain.

Data: openworm/c302 c302_A_Full.net.nml via the archived Cadence worm example
(cadence-examples/worm/data/connectome.json). Checks the wiring's identity and
structure, the cost of one sweep at this scale, and whether the library's
weighted contraction witness (a scalar h with |K|h <= alpha h) exists here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.brain import Brain
from cadence.learning import learning_neuron_model

DATA = Path("/Users/sheng/math/ep4nn/data/worm_connectome.json")
OUT: dict[str, object] = {}


def build() -> tuple[Connectome, dict]:
    raw = json.loads(DATA.read_text())
    names = [n["name"] for n in raw["neurons"]]
    index = {n: i for i, n in enumerate(names)}
    pre, post, count, sign = [], [], [], []
    for row in raw["chemical"]:
        a, b, c, s = int(row[0]), int(row[1]), float(row[2]), float(row[3])
        if a == b:
            continue
        pre.append(a)
        post.append(b)
        count.append(c)
        sign.append(s)
    for row in raw["gap"]:
        a, b, c = int(row[0]), int(row[1]), float(row[2])
        if a == b:
            continue
        for x, y in ((a, b), (b, a)):
            pre.append(x)
            post.append(y)
            count.append(c)
            sign.append(1.0)
    by_role: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for n in raw["neurons"]:
        by_role[n.get("role", "?")] = by_role.get(n.get("role", "?"), 0) + 1
        by_family[n.get("family", "?")] = by_family.get(n.get("family", "?"), 0) + 1
    populations = {
        "odour_A": tuple(index[n] for n in ("AWAL", "AWAR")),
        "odour_B": tuple(index[n] for n in ("AWCL", "AWCR")),
        "food": tuple(index[n] for n in ("CEPDL", "CEPDR", "CEPVL", "CEPVR",
                                         "ADEL", "ADER", "PDEL", "PDER")),
        "pain": tuple(index[n] for n in ("ASHL", "ASHR")),
        "forward": tuple(index[n] for n in ("AVBL", "AVBR", "PVCL", "PVCR")),
        "reverse": tuple(index[n] for n in ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")),
    }
    conn = Connectome.from_synapses(
        len(names), pre=pre, post=post, count=count, sign=sign, populations=populations,
        label="celegans:c302_A_Full",
    )
    meta = {
        "named_neurons": len(names),
        "roles": by_role,
        "families": by_family,
        "chemical_rows": len(raw["chemical"]),
        "gap_rows": len(raw["gap"]),
        "declared_inputs_present": {
            k: [names[i] for i in v] for k, v in populations.items()
        },
        "sources": raw["sources"],
    }
    return conn, meta


def main() -> None:
    conn, meta = build()
    OUT["data"] = meta
    OUT["graph"] = conn.summary()
    OUT["density"] = round(conn.synapses / conn.n**2, 4)
    OUT["self_synapses_dropped"] = True
    brain = Brain(conn, learning_neuron_model(), backend="cpu")
    OUT["neuron_model"] = {
        "gain": brain.neuron_model.gain,
        "dt": brain.neuron_model.dt,
        "blocked_dense": bool(brain._blocked),
        "layout_size": int(brain.layout.size),
    }
    drive = np.zeros((1, conn.n))
    drive[0, list(conn.populations["odour_A"])] = 1.0
    # library settle to the declared tolerance
    eq = brain.equilibrate(drive, budget=1024, chunk=32, tolerance=1e-4)
    OUT["settle"] = {
        "sweeps": int(eq.steps),
        "checks": int(eq.residual_checks),
        "residual": float(np.max(eq.residual)),
    }
    # one sweep cost with library transport vs the alternatives
    s = np.full((1, conn.n), 0.1)
    reps = 400
    from cadence.blocks import BlockTransport

    paths = {}
    if brain._blocks is not None:
        bt = BlockTransport(brain.layout, brain._blocks)
        bt.synaptic_input(s)
        t0 = time.perf_counter()
        for _ in range(reps):
            bt.synaptic_input(s)
        paths["blocked_dense_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    from cadence.sparse import transport

    csr = transport(conn, brain.weights)
    csr.dot(s.T).T
    t0 = time.perf_counter()
    for _ in range(reps):
        csr.dot(s.T).T
    paths["csr_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    brain._synaptic_input(s)
    t0 = time.perf_counter()
    for _ in range(reps):
        brain._synaptic_input(s)
    paths["library_path_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    t0 = time.perf_counter()
    for _ in range(reps):
        brain.settle_batch(drive, steps=1, state=None)
    paths["one_sweep_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    OUT["transport"] = paths

    # the paper's weighted contraction witness: K = L * |W|, look for h with K h <= alpha h
    W = brain.dense() if conn.n <= 2048 else None
    if W is not None:
        L = 0.125  # the activation Lipschitz bound the library's smooth rule declares
        K = np.abs(W)
        rho = float(np.max(np.abs(np.linalg.eigvals(K))))
        checks = {}
        for gain in (0.05, 0.1, 0.2, 1.0, float(brain.neuron_model.gain)):
            K = np.abs(W) * gain
            rad = float(np.max(np.abs(np.linalg.eigvals(K))))
            if rad < 1.0:
                h = np.ones(conn.n)
                for _ in range(200):
                    h = K @ h + 1.0
                alpha = float(np.max((K @ h) / h))
                checks[f"gain={gain:g}"] = {
                    "spectral_radius": round(rad, 4),
                    "witness_min_h": float(h.min()),
                    "witness_max_h": float(h.max()),
                    "alpha": round(alpha, 4),
                    "row_sum_abs_max": round(float(np.max(K.sum(axis=1))), 3),
                }
            else:
                checks[f"gain={gain:g}"] = {
                    "spectral_radius": round(rad, 4),
                    "row_sum_abs_max": round(float(np.max(K.sum(axis=1))), 3),
                    "certificate": "none (spectral radius >= 1)",
                }
        OUT["contraction"] = checks
        OUT["activation_lipschitz_used"] = L
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
