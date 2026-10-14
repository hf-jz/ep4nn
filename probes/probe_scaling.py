"""Probe 10: does Cadence scale past 302 neurons? Measured, not extrapolated.

Graphs with a fixed mean in-degree (the worm's ~10) at n = 302 (the real
C. elegans wiring, for the baseline), 1e3, 1e4, 1e5. Row-normalised weights so
the same numerical regime holds at every size (max |row sum| = 0.5), the
library's own rate model, then:
  * one 32-step chunk (what a settle costs, warm start)
  * sweeps to the declared tolerance from cold
  * memory of the CSR transport
  * the count the default constructors produce, for the density comparison
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
RHO = 0.5


def worm_connectome() -> Connectome:
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
                                    sign=sign, label="celegans")


def sparse_random(n: int, degree: float, seed: int = 4) -> Connectome:
    rng = np.random.default_rng(seed)
    edges = int(n * degree)
    pre = rng.integers(0, n, edges)
    post = rng.integers(0, n, edges)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    sign = rng.choice([-1.0, 1.0], size=len(pre), p=[0.12, 0.88])
    return Connectome.from_synapses(n, pre=pre, post=post, sign=sign, label=f"random{n}")


def closeness(conn: Connectome, rho: float = RHO) -> np.ndarray:
    """Efficacy that gives every neuron a |row sum| of rho (row-normalised wiring)."""
    indeg = np.bincount(conn.post, minlength=conn.n).astype(float)
    indeg[indeg == 0] = 1.0
    return conn.sign * (rho / indeg[conn.post])


def measure(conn: Connectome, label: str, reps: int = 5, chunk: int = 32) -> dict:
    brain = Brain(conn, learning_neuron_model(), backend="cpu",
                  efficacy=closeness(conn))
    n, nnz = conn.n, conn.synapses
    drive = np.zeros((1, n))
    drive[0, : min(8, n)] = 1.0
    warm = brain.settle_batch(drive, steps=chunk, state=None)
    t0 = time.perf_counter()
    for _ in range(reps):
        brain.settle_batch(drive, steps=chunk, state=warm)
    chunk_s = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    eq = brain.equilibrate(drive, budget=1024, chunk=chunk, tolerance=1e-4)
    settle_s = time.perf_counter() - t0
    from cadence.sparse import transport
    csr = transport(conn, brain.weights)
    bytes_ = None if csr is None else int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
    return {
        "label": label,
        "neurons": int(n),
        "synapses": int(nnz),
        "mean_in_degree": round(nnz / n, 2),
        "blocked_dense": bool(brain._blocked),
        "chunk32_ms": round(1000 * chunk_s, 3),
        "per_sweep_us": round(1e6 * chunk_s / chunk, 1),
        "cold_sweeps": int(eq.steps),
        "settle_ms": round(1000 * settle_s, 2),
        "residual": float(np.max(eq.residual)),
        "csr_MB": None if bytes_ is None else round(bytes_ / 1e6, 1),
    }


def constructor_density() -> list[dict]:
    from cadence import Brain as GenericBrain
    rows = []
    for modules in [(64, 32, 16), (128, 64, 32), (256, 128, 64), (512, 256, 128)]:
        b = GenericBrain.compose(inputs=16, actions=8, modules=modules, arousal=True, seed=1)
        core = b.brain
        rows.append({
            "modules": modules,
            "neurons": int(core.connectome.n),
            "synapses": int(core.connectome.synapses),
            "density": round(core.connectome.synapses / core.connectome.n ** 2, 4),
        })
    return rows


def main() -> None:
    OUT["worm_real"] = measure(worm_connectome(), "worm (real connectome)")
    for n, deg in ((1000, 10.4), (10000, 10.4), (100000, 12.4)):
        OUT[f"sparse_{n}"] = measure(sparse_random(n, deg), f"sparse random n={n}, degree~{deg}")
    OUT["default_constructors"] = constructor_density()
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
