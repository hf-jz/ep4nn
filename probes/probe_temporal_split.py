"""Is the 128-step memory bistability (two basins) or a slow decay (one continuum)?

§17b: after 128 blank steps the cue is still readable (sign accuracy 0.76). Two mechanisms fit:
  bistable -- different cues land in different attractor basins, so the final states split in two
  slow     -- one fixed point, but leakage is small enough that 128 steps do not arrive

Discriminator, computed on the final states alone:
  pc1_share      share of state variance in the first component (bistability -> large)
  sign_match     does sign(PC1) predict the cue's sign
  gap_ratio      between-group distance / mean within-group radius (>2 means two tight clusters,
                 ~1 means one blob)

    python probes/probe_temporal_split.py --trials 200 --k 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_manifold import build, load_worm_with_roles  # noqa: E402
from probe_temporal import trial_states  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    brain = build(conn, row_sum=0.5)
    states, y = trial_states(conn, brain, args.trials, args.k, args.seed)
    centred = states - states.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    pc1 = centred @ vt[0]
    var = s ** 2
    pc1_share = float(var[0] / var.sum())

    groups = np.sign(pc1)
    sign_match = float(np.mean(groups == np.sign(y)))
    if sign_match < 0.5:                       # PC1 sign is arbitrary
        groups, sign_match = -groups, 1.0 - sign_match

    between = float(abs(pc1[groups > 0].mean() - pc1[groups < 0].mean()))
    within = float(np.mean([pc1[groups == g].std() for g in (-1.0, 1.0)]))
    gap_ratio = between / within if within else float("inf")

    result = {"probe": "probe_temporal_split.py", "k": args.k, "trials": args.trials,
              "pc1_share": round(pc1_share, 3), "sign_match": round(sign_match, 3),
              "gap_ratio": round(gap_ratio, 2),
              "group_sizes": [int((groups > 0).sum()), int((groups < 0).sum())],
              "verdict": ("bistable: 两个紧吸引盆（记忆靠盆地，不靠时间常数）" if gap_ratio > 2
                          else "slow: 单一连续体（记忆靠小泄漏慢衰减）")}

    assert len(states) == args.trials
    assert 0.0 <= sign_match <= 1.0

    out = args.out or (HERE.parent / "docs" / "assets" / "temporal_split.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
