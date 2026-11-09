"""Step 1: the task the animal actually performs — forward vs reverse, not four-way stimulus ID.

Why: under the old nine-situation benchmark every 10-cell readout stayed inside a 1.3-degree cone,
and the reason is biological, not numerical — odour/food/pain all feed the same sensory-to-command
interneuron pathway, whose output is one locomotion decision. So measure that decision:

    L = mean(activation[forward: AVB/PVC]) - mean(activation[reverse: AVA/AVD/AVE])

which is the signed locomotion command the connectome is wired to produce (positive = forward).

Conditions carry an expected bias in [-1, +1] from the worm's known valence (attractants push
forward, repellents push reverse, mixtures interpolate). The test is whether L tracks that bias.

Pre-registered gate:
  PASS if Spearman rho(L, expected) >= 0.8 AND every sign agrees with the expected bias
  FAIL otherwise -> the connectome does not encode the locomotion command at this operating point,
  and the next lever is dynamics (slow variable), not the readout or growth.

    python probes/probe_locomotion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_manifold import build, collect, load_worm_with_roles  # noqa: E402

# (label, stimulus, expected bias)  -- bias is the worm's valence, fixed before running
CONDITIONS = [
    ("none", (), 0.0),
    ("attract_weak", (("odour_A", 0.25),), 0.25),
    ("attract_mid", (("odour_A", 0.50),), 0.50),
    ("attract_strong", (("odour_A", 1.00),), 1.00),
    ("food", (("food", 1.00),), 0.85),
    ("attractB", (("odour_B", 1.00),), 0.75),
    ("repel_weak", (("pain", 0.25),), -0.25),
    ("repel_mid", (("pain", 0.50),), -0.50),
    ("repel_strong", (("pain", 1.00),), -1.00),
    ("conflict_low", (("odour_A", 1.0), ("pain", 0.35)), 0.35),
    ("conflict_mid", (("odour_A", 1.0), ("pain", 0.70)), 0.0),
    ("conflict_high", (("odour_A", 1.0), ("pain", 1.00)), -0.40),
]


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--efficacy", choices=("row", "raw", "sqrt"), default="row")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    brain = build(conn, row_sum=args.row_sum, efficacy=args.efficacy)
    forward = np.array(sorted(conn.populations["forward"]))
    reverse = np.array(sorted(conn.populations["reverse"]))

    rows = []
    for label, senses, bias in CONDITIONS:
        drive = np.zeros((1, conn.n))
        for population, level in senses:
            drive[0, list(conn.populations[population])] = level
        state, sweeps, residual = collect(brain, drive)
        act = state[0]
        L = float(act[forward].mean() - act[reverse].mean())
        rows.append({"condition": label, "expected": bias, "L": round(L, 6),
                     "forward_mean": round(float(act[forward].mean()), 6),
                     "reverse_mean": round(float(act[reverse].mean()), 6),
                     "sweeps": sweeps, "residual": round(residual, 2)})

    Ls = [r["L"] for r in rows]
    bias = [r["expected"] for r in rows]
    rho = spearman(Ls, bias)
    signs = all((L > 0) == (b > 0) for L, b in zip(Ls, bias) if b != 0)
    passed = rho >= 0.8 and signs

    result = {"probe": "probe_locomotion.py", "row_sum": args.row_sum,
              "readout": "mean(AVB,PVC) - mean(AVA,AVD,AVE)",
              "spearman_rho": round(rho, 3), "signs_agree": signs,
              "L_range": round(max(Ls) - min(Ls), 6),
              "conditions": rows,
              "verdict": ("PASS: 连接组编码了 locomotion 指令（可按证据判增长）" if passed else
                          "FAIL: 该工作点下不编码 locomotion 极性 → 下一杠杆是动力学（慢变量）")}

    assert len(rows) == len(CONDITIONS)
    assert all(np.isfinite(r["L"]) for r in rows)

    out = args.out or (HERE.parent / "docs" / "assets" / "locomotion.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"rho={result['spearman_rho']} signs={signs} range={result['L_range']}")
    for r in rows:
        print(f"{r['condition']:14} expected={r['expected']:+.2f} L={r['L']:+.5f} "
              f"fwd={r['forward_mean']:.4f} rev={r['reverse_mean']:.4f}")
    print(result["verdict"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
