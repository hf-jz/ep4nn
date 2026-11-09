"""A2: re-pick the readout — by situation separation, not by raw variance.

Two corrections after the first run (2026-10-20):
  1. "share of variance" must be additive: sum(var[cells]) / sum(var[all]) <= 1. The first
     version divided two means and printed 6.8 / 13.5, which is meaningless.
  2. maximising variance picks the sensory cells (AWCL/PDER/ASH...), i.e. the readout would be
     reading the drive, not the computation. Those populations are excluded from the candidate
     pool, and the objective becomes separation of the canonical situations.

Objective: greedy forward selection maximising the minimum pairwise cosine distance between the
nine situation responses, so every pair must be distinguishable and not just the average.

Pre-registered gate: PASS if min_pair_distance >= 0.5 (worst pair at most 60 deg apart) with 10
non-sensory cells; FAIL otherwise -> readout search is closed, the bottleneck is upstream.

    python probes/probe_readout_search.py --samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_manifold import (build, collect, drives, drives_sensory,  # noqa: E402
                            load_worm_with_roles, spectrum)
from probe_readout_response import SITUATIONS  # noqa: E402

DECLARED = ("AVBL", "AVBR", "PVCL", "PVCR", "AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")
SENSORY_POPS = ("odour_A", "odour_B", "food", "pain")


def pair_distance(resp: np.ndarray) -> float:
    """Smallest cosine distance over all pairs of situation responses (rows)."""
    n = resp.shape[0]
    norms = np.linalg.norm(resp, axis=1)
    norms[norms == 0] = 1.0
    unit = resp / norms[:, None]
    dists = [1.0 - abs(float(unit[i] @ unit[j])) for i in range(n) for j in range(i + 1, n)]
    return float(min(dists)) if dists else 0.0


def situation_responses(brain, conn) -> np.ndarray:
    rows = []
    for senses in SITUATIONS:
        drive = np.zeros((1, conn.n))
        for population, level in senses:
            drive[0, list(conn.populations[population])] = level
        state, _, _ = collect(brain, drive)
        rows.append(state[0])
    return np.stack(rows)


def eval_set(resp: np.ndarray, var: np.ndarray, idx: np.ndarray, with_pairs: bool = True) -> dict:
    ro = resp[:, idx]
    ro_c = ro - ro.mean(axis=0)
    out = {"share_of_var": round(float(var[idx].sum() / var.sum()), 4),
           "participation_ratio": spectrum(ro_c)["participation_ratio"]}
    if with_pairs:
        out["min_pair_distance"] = round(pair_distance(ro_c), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--input-mode", choices=("pops", "sensory"), default="pops")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    names = [c["name"] for c in json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "worm_connectome.json").read_text())["neurons"]]
    brain = build(conn, row_sum=args.row_sum)

    if args.input_mode == "sensory":
        x, input_dim = drives_sensory(conn, args.samples, args.seed)
    else:
        x, input_dim = drives(conn, args.samples, args.seed), 4
    state, sweeps, residual = collect(brain, x)
    var = (state - state.mean(axis=0)).var(axis=0)

    resp = situation_responses(brain, conn) if args.input_mode == "pops" else state
    resp_c = resp - resp.mean(axis=0)
    with_pairs = args.input_mode == "pops"

    excluded: set[int] = set()
    for population in SENSORY_POPS:
        excluded |= set(conn.populations[population])
    pool = [i for i in range(conn.n) if i not in excluded]
    declared = np.array([names.index(n) for n in DECLARED])

    picked: list[int] = []
    for _ in range(args.k):
        best, best_score = None, -1.0
        for cell in pool:
            if cell in picked:
                continue
            if with_pairs:
                score = pair_distance(resp_c[:, picked + [cell]])
            else:
                score = spectrum(resp_c[:, picked + [cell]])["participation_ratio"]
            if score > best_score:
                best, best_score = cell, score
        picked.append(best)

    result = {
        "probe": "probe_readout_search.py", "row_sum": args.row_sum, "samples": args.samples,
        "sweeps_max": sweeps, "residual_max": round(residual, 2),
        "pool": len(pool), "k": args.k,
        "declared": {**eval_set(resp, var, declared), "cells": list(DECLARED)},
        "greedy": {**eval_set(resp, var, np.array(picked)), "cells": [names[i] for i in picked]},
        "ceiling_share_of_var": round(float(np.sort(var)[-args.k:].sum() / var.sum()), 4),
    }
    if with_pairs:
        passed = result["greedy"]["min_pair_distance"] >= 0.5
        result["verdict"] = ("PASS: 换读出值得做（最差一对也能分开）" if passed else
                             "FAIL: 10 个非感觉细胞仍分不开情境 → 读出搜索关闭，瓶颈在输入维数/架构")
    else:
        passed = (result["greedy"]["share_of_var"] >= 0.20
                  and result["greedy"]["participation_ratio"] >= 3.0)
        result["verdict"] = ("PASS: 高维输入下换读出值得做" if passed else
                             "FAIL: 高维输入下 10 个细胞也撑不起读出 → 瓶颈在 302 图本身的读出层")

    assert len(picked) == args.k and len(set(picked)) == args.k
    assert result["greedy"]["share_of_var"] <= 1.0 and result["declared"]["share_of_var"] <= 1.0

    out = args.out or (Path(__file__).resolve().parents[1] / "docs" / "assets" / "readout_search.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"verdict": result["verdict"]}, ensure_ascii=False))
    for arm in ("declared", "greedy"):
        r = result[arm]
        print(f"{arm:9} share={r['share_of_var']:<7} PR={r['participation_ratio']:<7} "
              f"min_dist={r.get('min_pair_distance')}  {r['cells']}")
    print(f"ceiling share (any 10 cells, incl. sensory) = {result['ceiling_share_of_var']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
