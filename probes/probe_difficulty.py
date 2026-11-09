"""A4: the first scaling curve — does task difficulty raise the readout rank a task needs?

§15 showed a rank-1 readout already gets the binary decision right, so nothing needed to grow.
The only way difficulty can demand more rank is *interference*: the label stays a function of a
few informative cells while the state is also driven by an increasing number of irrelevant ones.
A readout then has to project the nuisance out, which costs rank.

Difficulty axis d = number of nuisance sensory cells driven with independent random levels
(0, 4, 8, 16, 32). Label stays v = tanh(3 (a - r)) over the same 14 informative cells.
Everything else is inherited from probe_mapping_layer (PCA-truncated ridge, 5-fold CV).

Pre-registered judgement:
  required rank = smallest r whose held-out rho is within 0.02 of the best rho at that d.
  PASS (scaling exists) if rank_needed(d=32) - rank_needed(d=0) >= 4.
  FAIL if the curve stays flat (<= 2): difficulty of this kind does not require capacity, and
  the adaptive-growth hypothesis has no support on this substrate.

    python probes/probe_difficulty.py --samples 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_mapping_layer import fit_predict, spearman  # noqa: E402
from probe_manifold import build, collect, load_worm_with_roles, sensory_cells  # noqa: E402

RANKS = (1, 2, 4, 8, 16, 32, 64)
NOVELTY_LEVELS = (0, 4, 8, 16, 32)
INFORMATIVE_POPS = ("odour_A", "odour_B", "food", "pain")


def make_batch(conn, samples: int, nuisance: int, seed: int):
    """Informative drives on the 4 named populations + `nuisance` random cells driven independently."""
    rng = np.random.default_rng(seed)
    informative = np.concatenate([conn.populations[p] for p in INFORMATIVE_POPS])
    distractors = np.array([c for c in sensory_cells(conn) if c not in set(informative)])
    chosen = rng.choice(distractors, size=min(nuisance, len(distractors)), replace=False)

    x = np.zeros((samples, conn.n))
    for i in range(samples):                     # valence 由 4 个命名群体决定
        for population in INFORMATIVE_POPS:
            if rng.random() < 0.6:
                x[i, list(conn.populations[population])] = rng.uniform(0.05, 1.0)
        if len(chosen):                          # nuisance：与标签独立
            x[i, chosen] = rng.uniform(0.05, 1.0, len(chosen))
    attract = np.concatenate([conn.populations[p] for p in ("odour_A", "odour_B", "food")])
    repel = np.array(sorted(conn.populations["pain"]))
    y = np.tanh(3.0 * (x[:, attract].sum(axis=1) - x[:, repel].sum(axis=1)))
    return x, y, len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    brain = build(conn, row_sum=args.row_sum)
    curve = []
    for nuisance in NOVELTY_LEVELS:
        x, y, used = make_batch(conn, args.samples, nuisance, args.seed)
        states, sweeps, _ = collect(brain, x)
        order = np.random.default_rng(args.seed).permutation(len(states))
        folds = np.array_split(order, args.folds)
        rows = []
        for rank in RANKS:
            rhos = []
            for test in folds:
                mask = np.ones(len(states), bool)
                mask[test] = False
                rhos.append(spearman(fit_predict(states[mask], y[mask], states[test], rank), y[test]))
            rows.append({"rank": rank, "rho": round(float(np.mean(rhos)), 3)})
        best = max(r["rho"] for r in rows)
        needed = next(r["rank"] for r in rows if r["rho"] >= best - 0.02)
        curve.append({"nuisance_cells": used, "rank_needed": needed, "best_rho": best,
                      "rows": rows, "sweeps": sweeps})
        print(f"nuisance={used:<3} rank_needed={needed:<3} best_rho={best} "
              f" rho@1={rows[0]['rho']} rho@8={rows[3]['rho']}", flush=True)

    delta = curve[-1]["rank_needed"] - curve[0]["rank_needed"]
    result = {"probe": "probe_difficulty.py", "samples": args.samples, "row_sum": args.row_sum,
              "levels": list(NOVELTY_LEVELS), "curve": curve, "rank_delta": delta,
              "verdict": ("PASS: 难度（干扰）确实抬高所需秩，scaling 曲线成立" if delta >= 4 else
                          "FAIL: 曲线平坦，干扰不抬高所需秩 → 该基质上不支持自适应扩规模")}
    assert len(curve) == len(NOVELTY_LEVELS)
    assert all(r["rank_needed"] in RANKS for r in curve)

    out = args.out or (HERE.parent / "docs" / "assets" / "difficulty_curve.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"rank_delta": delta, "verdict": result["verdict"]}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
