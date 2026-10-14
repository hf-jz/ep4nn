"""Probe 6: where the worm-scale settle time actually goes (blocked dense vs CSR)."""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.brain import Brain, _FUSED
from cadence.learning import learning_neuron_model
from cadence.sparse import transport


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


def bench(conn, dense_limit, steps=32, reps=200):
    brain = Brain(conn, learning_neuron_model(), backend="cpu", dense_limit=dense_limit)
    drive = np.zeros((1, conn.n))
    drive[0, :4] = 1.0
    brain.settle_batch(drive, steps=steps, state=None)
    t0 = time.perf_counter()
    for _ in range(reps):
        brain.settle_batch(drive, steps=steps, state=None)
    per_sweep = 1e6 * (time.perf_counter() - t0) / (reps * steps)
    from cadence.blocks import BlockTransport

    blocks = None if brain._blocks is None else BlockTransport(brain.layout, brain._blocks)
    s = np.full((1, conn.n), 0.1)
    t0 = time.perf_counter()
    for _ in range(2000):
        brain._synaptic_input(s, blocks)
    transport_us = 1e6 * (time.perf_counter() - t0) / 2000
    return {
        "dense_limit": dense_limit,
        "blocked": bool(brain._blocked),
        "layout_entries": int(brain.layout.size),
        "transport_us": round(transport_us, 1),
        "sweep_us": round(per_sweep, 1),
    }


def profile(conn, dense_limit, reps=30):
    brain = Brain(conn, learning_neuron_model(), backend="cpu", dense_limit=dense_limit)
    drive = np.zeros((1, conn.n))
    drive[0, :4] = 1.0
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(reps):
        brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(12)
    return buf.getvalue()


def main() -> None:
    conn = build()
    out = {"fused_available": bool(_FUSED), "csr_available": transport(conn, np.ones(conn.synapses)) is not None}
    out["blocked_default"] = bench(conn, 2048)
    out["csr_forced"] = bench(conn, 0)
    print(json.dumps(out, indent=2))
    print("=== profile blocked (top) ===")
    print(profile(conn, 2048))
    print("=== profile CSR (top) ===")
    print(profile(conn, 0))


if __name__ == "__main__":
    main()
