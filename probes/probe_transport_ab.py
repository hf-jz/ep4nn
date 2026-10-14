"""Probe 8: resolve the end-to-end difference between the block-dense and CSR paths.

Times, for the same 302-neuron graph and the same drive:
  * one 32-step chunk   settle_batch(drive, steps=32, state=warm)
  * one residual        residual(drive, state)
  * one sweep with a cold state (steps=1, state=None)
  * one whole equilibrate
under both transport selections, plus the raw transport call each one takes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.blocks import BlockTransport
from cadence.brain import Brain, _FUSED
from cadence.learning import learning_neuron_model
from cadence.sparse import transport

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


def timed(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return round(1e6 * (time.perf_counter() - t0) / reps, 1)


def measure(conn: Connectome, dense_limit: int, reps: int = 300) -> dict:
    brain = Brain(conn, learning_neuron_model(), backend="cpu", dense_limit=dense_limit)
    drive = np.zeros((1, conn.n))
    drive[0, :4] = 1.0
    warm = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4).state
    blocks = None if brain._blocks is None else BlockTransport(brain.layout, brain._blocks)
    s = np.full((1, conn.n), 0.1)
    csr = transport(conn, brain.weights)
    out = {
        "dense_limit": dense_limit,
        "blocked": bool(brain._blocked),
        "fused_available": bool(_FUSED),
        "transport_call_us": round(timed(lambda: brain._synaptic_input(s, blocks), 2000), 1),
        "csr_call_us": None if csr is None else round(timed(lambda: csr.dot(s.T).T, 2000), 1),
        "residual_us": round(timed(lambda: brain.residual(drive, warm), 300), 1),
        "chunk32_us": round(timed(lambda: brain.settle_batch(drive, steps=32, state=warm), reps), 1),
        "cold_1step_us": round(timed(lambda: brain.settle_batch(drive, steps=1, state=None), reps), 1),
        "equilibrate_us": round(timed(
            lambda: brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4), 50), 1),
        "equilibrate_sweeps": int(brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4).steps),
    }
    out["chunk32_per_sweep_us"] = round(out["chunk32_us"] / 32, 2)
    return out


def main() -> None:
    conn = build()
    OUT["blocked"] = measure(conn, 2048)
    OUT["csr"] = measure(conn, 0)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
