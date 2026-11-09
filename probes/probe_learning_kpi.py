"""P2: the two learning KPIs from Wojcik et al. 2026, applied to the frozen substrate.

That paper reports what learning does to prefrontal geometry: task-relevant relations stay
decodable, irrelevant features become harder to decode, and the structure transfers to a new
stimulus set. Accuracy alone cannot show any of that. So before we can ask "does growth buy
learning", we need to know whether the frozen substrate already supports a transferable valence
geometry, or whether it only carries cell identity.

Two disjoint chemosensory sets (odour_A, odour_B), same label rule for both:
    appetitive pattern (drive on the set's odour cells) -> label +intensity
    aversive pattern   (drive on the pain cells)        -> label -intensity
Three numbers:
    rho_valence   held-out rank correlation within a set (5-fold)
    rho_set_id    how decodable the *set* is from the same states (the nuisance feature)
    rho_transfer  train on A, test on B and vice versa (zero-shot, no fitting on B)

Pre-registered gate: the substrate supports abstract valence if transfer rho >= 0.5. If transfer
collapses while set identity stays decodable, the state encodes which cells fired, not what they
mean, and no readout-side learning can fix that.

    python probes/probe_learning_kpi.py --samples 200
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
from probe_mapping_layer import fit_predict, spearman  # noqa: E402

RANK = 8


def pattern_set(conn, cells, pain, n, rng):
    rows, labels = [], []
    for _ in range(n):
        row = np.zeros(conn.n)
        intensity = float(rng.uniform(0.3, 1.0))
        positive = bool(rng.integers(2))
        pool = np.asarray(cells if positive else pain)
        k = int(rng.integers(2, max(3, len(pool) // 2)))
        row[rng.choice(pool, size=k, replace=False)] = intensity
        rows.append(row)
        labels.append(intensity if positive else -intensity)
    return np.stack(rows), np.array(labels)


def cv_rho(states, y, folds):
    rhos = []
    for test in folds:
        mask = np.ones(len(states), bool)
        mask[test] = False
        rhos.append(spearman(fit_predict(states[mask], y[mask], states[test], RANK), y[test]))
    return float(np.mean(rhos))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    brain = build(conn, row_sum=args.row_sum)
    pain = np.array(sorted(conn.populations["pain"]))
    set_a = np.array(sorted(conn.populations["odour_A"]))
    set_b = np.array(sorted(conn.populations["odour_B"]))

    rng = np.random.default_rng(args.seed)
    xa, ya = pattern_set(conn, set_a, pain, args.samples, rng)
    xb, yb = pattern_set(conn, set_b, pain, args.samples, rng)
    sa, _, _ = collect(brain, xa)
    sb, _, _ = collect(brain, xb)

    folds_a = np.array_split(rng.permutation(len(sa)), args.folds)
    folds_b = np.array_split(rng.permutation(len(sb)), args.folds)
    within = {name: cv_rho(s, y, f) for name, s, y, f in
              (("A", sa, ya, folds_a), ("B", sb, yb, folds_b))}

    transfer = {}
    for name, src_s, src_y, dst_s, dst_y in (("A->B", sa, ya, sb, yb), ("B->A", sb, yb, sa, ya)):
        pred = fit_predict(src_s, src_y, dst_s, RANK)
        transfer[name] = round(spearman(pred, dst_y), 3)

    pooled = np.concatenate([sa, sb])
    set_id = np.concatenate([np.zeros(len(sa)), np.ones(len(sb))])
    fold_pool = np.array_split(rng.permutation(len(pooled)), args.folds)
    rho_set = float(np.mean([
        np.mean((fit_predict(pooled[m], set_id[m], pooled[t], RANK) > 0.5) == (set_id[t] > 0.5))
        for t in fold_pool for m in [np.setdiff1d(np.arange(len(pooled)), t)]]))
    rho_set = 2.0 * (rho_set - 0.5)                       # chance corrected: 0 = no information

    rho_valence = float(np.mean(list(within.values())))
    rho_transfer = float(np.mean(list(transfer.values())))
    result = {"probe": "probe_learning_kpi.py", "samples": args.samples, "rank": RANK,
              "rho_valence_within": {k: round(v, 3) for k, v in within.items()},
              "rho_transfer": transfer,
              "rho_set_id_chance_corrected": round(rho_set, 3),
              "ratio_valence_over_set_id": round(rho_valence / max(rho_set, 1e-3), 2)}
    passed = rho_transfer >= 0.5
    result["verdict"] = ("PASS: 状态带抽象效价，跨刺激集可迁移" if passed else
                         "FAIL: 状态携带的是细胞身份，不是效价含义；换读出/加学习都救不了")

    assert len(ya) == len(yb) == args.samples
    assert 0.0 <= rho_set <= 1.0

    out = args.out or (HERE.parent / "docs" / "assets" / "learning_kpi.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"valence within: {result['rho_valence_within']}  mean {rho_valence:.3f}")
    print(f"transfer: {transfer}  mean {rho_transfer:.3f}")
    print(f"set-id decodability (chance corrected): {rho_set:.3f}")
    print(f"ratio valence/set-id: {result['ratio_valence_over_set_id']}")
    print(result["verdict"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
