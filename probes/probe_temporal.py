"""A5: the last untested axis — does a delayed cue survive in this substrate at all?

Every earlier probe settled each drive to its own fixed point, i.e. the harness had no temporal
state. So "difficulty = temporal depth" was never actually tested. This one carries BrainState
across time steps:

    t=0   cue (odour_A or pain, random level)   1 step
    t>0   blank drive                           1 step each, repeated k times
    read  the state at t=k, predict the cue's sign and magnitude

k = 0..16. Readout identical to probe_mapping_layer (PCA + ridge, 5-fold CV over trials).
Also runs with spike-frequency adaptation (slow variable) to see whether a slower neuron state
extends the horizon.

Pre-registered judgement:
  memory horizon k* = largest k with held-out sign accuracy >= 0.75.
  PASS (temporal axis exists) if k* >= 4 at some adaptation setting.
  FAIL if accuracy is at chance already at k=1 in every setting: this substrate cannot carry a
  cue across a single blank step, so "more neurons" is the wrong lever and an explicit memory
  structure is required.

    python probes/probe_temporal.py --trials 400
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_mapping_layer import fit_predict, spearman  # noqa: E402
from probe_manifold import build, load_worm_with_roles  # noqa: E402

from cadence.neuron import Adaptation  # noqa: E402

STEPS = (0, 1, 2, 4, 8, 16)
RANKS = (1, 4, 16)


def trial_states(conn, brain, trials: int, k: int, seed: int):
    """Cue at t=0, k blank steps, return the states at t=k with their labels."""
    rng = np.random.default_rng(seed + 1000 * k)
    attract = list(conn.populations["odour_A"])
    repel = list(conn.populations["pain"])
    states, labels = [], []
    for _ in range(trials):
        sign = 1.0 if rng.random() < 0.5 else -1.0
        cells = attract if sign > 0 else repel
        level = float(rng.uniform(0.2, 1.0))
        cue = np.zeros((1, conn.n))
        cue[0, cells] = level
        state = None
        for step in range(k + 1):
            drive = cue if step == 0 else np.zeros((1, conn.n))
            state = brain.settle_batch(drive, steps=1, state=state)
        states.append(np.asarray(state.activation).ravel())
        labels.append(sign * level)
    return np.stack(states), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS),
                        help="delays to test; use e.g. 0 4 16 32 64 128 for the failure point")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    settings = {"none": None, "adapt_tau50": Adaptation(tau_steps=50.0, strength=1.0),
                "adapt_tau200": Adaptation(tau_steps=200.0, strength=1.0)}
    result = {"probe": "probe_temporal.py", "trials": args.trials, "row_sum": args.row_sum,
              "steps": list(args.steps), "settings": {}}

    for name, adaptation in settings.items():
        brain = build(conn, row_sum=args.row_sum, adaptation=adaptation)
        rows = []
        for k in args.steps:
            states, y = trial_states(conn, brain, args.trials, k, args.seed)
            order = np.random.default_rng(args.seed).permutation(len(states))
            folds = np.array_split(order, args.folds)
            best = {"rank": None, "acc": 0.0, "rho": 0.0}
            for rank in RANKS:
                accs, rhos = [], []
                for test in folds:
                    mask = np.ones(len(states), bool)
                    mask[test] = False
                    pred = fit_predict(states[mask], y[mask], states[test], rank)
                    accs.append(float(np.mean(np.sign(pred) == np.sign(y[test]))))
                    rhos.append(spearman(pred, y[test]))
                if float(np.mean(accs)) > best["acc"]:
                    best = {"rank": rank, "acc": round(float(np.mean(accs)), 3),
                            "rho": round(float(np.mean(rhos)), 3)}
            rows.append({"k": k, **best})
            print(f"{name:12} k={k:<3} sign_acc={best['acc']:.3f} rho={best['rho']:.3f} "
                  f"(rank {best['rank']})", flush=True)
        horizon = max((r["k"] for r in rows if r["acc"] >= 0.75), default=-1)
        result["settings"][name] = {"rows": rows, "horizon": horizon}

    best_horizon = max(s["horizon"] for s in result["settings"].values())
    result["best_horizon"] = best_horizon
    result["verdict"] = ("PASS: 时间轴存在（线索可跨 ≥4 步）" if best_horizon >= 4 else
                         "FAIL: 线索连一个空白步都撑不过 → 需要显式记忆结构，而不是加神经元")

    assert len(result["settings"]) == len(settings)
    assert all(0.0 <= r["acc"] <= 1.0 for s in result["settings"].values() for r in s["rows"])

    out = args.out or (HERE.parent / "docs" / "assets" / "temporal.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"best_horizon": best_horizon, "verdict": result["verdict"]},
                     ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
