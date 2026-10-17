"""Is "double the neurons" a capability claim or a counter?

Capacity is cheap to measure and the paper's negative result makes the distinction sharp: adding
neurons is a count, participating in the answer is a measurement. This forces growth to 2x and 4x the
worm's 302 neurons without the verdict layer (the verdict is about benefit, this is about capacity),
and reports what each doubling buys: neurons, synapses, cost, the share of cells that fire at all, the
share that can reach the readout, and whether the declared answer moves.

It also builds every composed brain the library's own examples use, so the two sides are compared at
their real sizes rather than at their descriptions.

    python probes/probe_capacity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ep4nn.grow import EP4NN, GrowthConfig, load_worm, operating_point   # noqa: E402
from cadence import Brain as ComposedBrain                            # noqa: E402
from cadence.brain import Brain                                        # noqa: E402

OUT: dict[str, object] = {}
TARGETS = (302, 604, 1208)


def reach(ep: EP4NN) -> float:
    """Share of neurons with a directed path of at most three edges to the declared readout."""
    n = ep.neurons
    frontier = np.zeros(n, dtype=bool)
    frontier[ep.readout] = True
    seen = frontier.copy()
    for _ in range(3):
        nxt = np.isin(ep.connectome.post, np.flatnonzero(frontier))
        frontier = np.zeros(n, dtype=bool)
        frontier[ep.connectome.pre[nxt]] = True
        seen |= frontier
    return float(seen.mean())


def measure(ep: EP4NN) -> dict:
    codes = ep._situation_codes()
    half = len(codes) // 2
    norms = np.linalg.norm(codes, axis=1)
    coll = 1.0 if norms.min() == 0 else max(
        abs(float(codes[i] @ codes[j]) / (norms[i] * norms[j]))
        for s in (0, half) for i in range(s, s + half) for j in range(i + 1, s + half))
    drive = ep._drive(("odour_A", 1.0), ("pain", 1.0))
    warm = ep.brain.settle_batch(drive, steps=ep.chunk, state=None)
    t0 = time.perf_counter()
    for _ in range(5):
        ep.brain.settle_batch(drive, steps=ep.chunk, state=warm)
    per_sweep = 1e6 * (time.perf_counter() - t0) / (5 * ep.chunk)
    t0 = time.perf_counter()
    eq = ep.brain.equilibrate(drive, budget=1024, chunk=ep.chunk, tolerance=1e-4)
    settle_ms = 1000 * (time.perf_counter() - t0)
    state = np.asarray(eq.state.activation).ravel()
    return {"neurons": ep.neurons, "synapses": ep.synapses,
            "edges_per_neuron": round(ep.synapses / ep.neurons, 2),
            "per_sweep_us": round(per_sweep, 1), "settle_ms": round(settle_ms, 2),
            "sweeps": int(eq.steps),
            "firing_fraction": round(float((state > 1e-9).mean()), 4),
            "reachable_to_readout": round(reach(ep), 4),
            "readout_spread": round(float(np.mean(codes.max(axis=0) - codes.min(axis=0))), 6),
            "collision": round(float(coll), 6)}


def grow_to(ep: EP4NN, target: int, codes: list[np.ndarray]) -> int:
    """Append neurons until the target is reached. No verdict: this asks what doubling costs."""
    confused = [(codes[0], 0, 0), (codes[-1], 0, len(codes) - 1)]
    events = 0
    while ep.neurons < target and events < 200:
        ep.ticks = events
        added = ep._grow(confused, np.random.default_rng(100 + events))
        if not added:
            break
        ep._rebuild()
        events += 1
    return events


def composed() -> list[dict]:
    """The composed brains the library's examples actually build."""
    rows = []
    for label, kwargs in (
        ("continuing_brain: compose(2, 2, modules=(8,))", dict(inputs=2, actions=2, modules=(8,))),
        ("walking_for_reward: compose(..., modules=(32,))", dict(inputs=2, actions=2, modules=(32,))),
        ("walking_rhythm: compose(..., modules=(32,))", dict(inputs=2, actions=2, modules=(32,))),
        ("memory_imagination: compose(..., modules=(32,))", dict(inputs=2, actions=2, modules=(32,))),
    ):
        b = ComposedBrain.compose(backend="cpu", seed=7, **kwargs)
        core = b.brain
        rows.append({"example": label, "neurons": int(core.connectome.n),
                     "synapses": int(core.connectome.synapses)})
    return rows


def main() -> None:
    conn = load_worm()
    arms = (("as the generator is", GrowthConfig(calibrate=False)),
            ("density held", GrowthConfig(calibrate=False, degree=4.5, to_readout=0.4)),
            ("readout barred", GrowthConfig(calibrate=False, degree=4.5, to_readout=0.0)),
            ("operating point held", GrowthConfig(calibrate=False, degree=4.5, to_readout=0.4,
                                                  init_scale=1.0)),
            ("held, readout barred", GrowthConfig(calibrate=False, degree=4.5, to_readout=0.0,
                                                  init_scale=1.0)))
    for label, cfg in arms:
        ep = EP4NN(load_worm(), cfg)
        codes = list(ep._situation_codes())
        rows = [measure(ep)]
        for target in TARGETS[1:]:
            events = grow_to(ep, target, codes)
            rows.append({**measure(ep), "growth_events": events})
        OUT[label] = rows
    OUT["composed_examples"] = composed()
    print("worm graph, forced growth (no verdict: capacity only)")
    print(" arm              neurons synapses e/n   us/sweep settle_ms sweeps firing  spread   collision")
    for label in [k for k, v in OUT.items() if k not in ("composed_examples",)]:
        rows = OUT[label]
        for r in rows:
            print(f" {label:<16} {r['neurons']:>7} {r['synapses']:>8} {r['edges_per_neuron']:>4.1f} "
                  f"{r['per_sweep_us']:>9.1f} {r['settle_ms']:>9.2f} {r['sweeps']:>6} "
                  f"{r['firing_fraction']:>6.3f} {r['readout_spread']:>8.5f} {r['collision']:>9.6f}")
    print("\ncomposed brains the library's examples build")
    for r in OUT["composed_examples"]:
        print(f" {r['example']:<55} {r['neurons']:>6} neurons {r['synapses']:>7} synapses")
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "capacity.json"
    dump.write_text(json.dumps(OUT, indent=1) + "\n")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
