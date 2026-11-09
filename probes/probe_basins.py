"""Does scale buy memory slots? Basin count vs connectome size, on the genome/develop path.

§17c: memory is which attractor basin the state lands in, not a decaying trace. So the only
quantity growth could plausibly buy is **how many cues stay distinguishable after a long blank
delay**. This measures exactly that, on a structure family (the same genome, only the hidden
region's size changes — §"结构族参数化"):

    input(14) -> hidden(k) -> output(10), fixed densities and signs, k = 32/64/128/256
    cue at t=0, 64 blank steps, then: are the n_cues cues still separable?

Separability = min over cue pairs of ||mean_i - mean_j|| / mean within-cue spread.
Read as: how many cues the substrate can hold at that size.

Pre-registered judgement:
  PASS (scale buys slots) if the largest size holds >= 2x the cues of the smallest.
  FAIL if the count is flat across a 8x size range: basin count is set by topology, not scale.

    python probes/probe_basins.py --sizes 32 64 128 256 --delay 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_manifold import load_worm_with_roles  # noqa: E402

from cadence.brain import Brain  # noqa: E402
from cadence.genome import Genome, Projection, Region, develop  # noqa: E402
from cadence.learning import learning_neuron_model  # noqa: E402

CUE_COUNTS = (4, 8, 16, 32, 64)
DENSITY = 0.15


def genome(hidden: int, inputs: int = 14, outputs: int = 10, seed: int = 0) -> Genome:
    regions = (Region("input", size=inputs), Region("hidden", size=hidden),
               Region("output", size=outputs))
    projections = (Projection("input", "hidden", density=DENSITY, sign=0.6, scale=1.0),
                   Projection("hidden", "hidden", density=DENSITY, sign=0.0, scale=1.0),
                   Projection("hidden", "output", density=DENSITY, sign=0.6, scale=1.0))
    return Genome(regions=regions, projections=projections, label=f"family-{hidden}")


def run_size(hidden: int, cues: int, delay: int, trials: int, seed: int,
             input_div: float = 0.0) -> dict:
    inputs = max(14, int(round(hidden / input_div))) if input_div else 14
    conn = develop(genome(hidden, inputs=inputs, seed=seed), seed=seed)
    brain = Brain(conn, learning_neuron_model(), backend="cpu")
    input_cells = np.array(sorted(conn.populations["input"]))
    rng = np.random.default_rng(seed + 17 * hidden)
    # overlapping random patterns, not disjoint slices: disjoint subsets cap the number of cues
    # at the input size and would hide the effect we are looking for
    patterns = rng.uniform(0.1, 1.0, (cues, len(input_cells)))
    states, labels = [], []
    for cue in range(cues):
        for _ in range(trials):
            drive = np.zeros((1, conn.n))
            drive[0, input_cells] = patterns[cue] * rng.uniform(0.9, 1.1, len(input_cells))
            state = None
            for step in range(delay + 1):
                state = brain.settle_batch(drive if step == 0 else np.zeros((1, conn.n)),
                                           steps=1, state=state)
            states.append(np.asarray(state.activation).ravel())
            labels.append(cue)
    x = np.stack(states)
    y = np.array(labels)

    means = np.stack([x[y == c].mean(axis=0) for c in range(cues)])
    spread = float(np.mean([x[y == c].std(axis=0).mean() for c in range(cues)]))
    pair = [float(np.linalg.norm(means[i] - means[j])) / (spread + 1e-12)
            for i in range(cues) for j in range(i + 1, cues)]
    return {"hidden": hidden, "n_neurons": conn.n, "cues": cues,
            "min_pair_separation": round(min(pair), 2) if pair else None,
            "separable": bool(pair and min(pair) >= 2.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--delay", type=int, default=64)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-div", type=float, default=0.0,
                        help="if set, input width = hidden/input-div (e.g. 4 -> input = hidden/4)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = []
    for size in args.sizes:
        for cues in CUE_COUNTS:
            row = run_size(size, cues, args.delay, args.trials, args.seed, args.input_div)
            rows.append(row)
            print(f"hidden={size:<4} neurons={row['n_neurons']:<4} cues={cues:<2} "
                  f"min_sep={row['min_pair_separation']:<6} separable={row['separable']}",
                  flush=True)

    def held(size: int) -> int:
        return max((r["cues"] for r in rows if r["hidden"] == size and r["separable"]), default=0)

    small, large = held(args.sizes[0]), held(args.sizes[-1])
    result = {"probe": "probe_basins.py", "delay": args.delay, "trials": args.trials,
              "rows": rows, "held_small": small, "held_large": large,
              "verdict": (f"PASS: 规模买到记忆槽位（{small} → {large}）" if large >= max(2, 2 * small)
                          else f"FAIL: 槽位数与规模无关（{small} → {large}），由拓扑决定")}

    assert len(rows) == len(args.sizes) * len(CUE_COUNTS)
    out = args.out or (HERE.parent / "docs" / "assets" / "basins.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"held_small": small, "held_large": large, "verdict": result["verdict"]},
                     ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
