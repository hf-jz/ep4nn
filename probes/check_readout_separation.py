"""Does the searched (10-cell, high-dimensional) readout separate the nine canonical situations?

The declared readout gives min pair cosine distance 0.005 (answers are parallel). The sensory-mode
search found a 10-cell set whose effective dimensionality is 8.51 vs the declared 1.88. This asks
the operational question: with that set, are the nine situations actually distinguishable?

    python probes/check_readout_separation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_manifold import build, collect, load_worm_with_roles  # noqa: E402
from probe_readout_response import SITUATIONS  # noqa: E402

DECLARED = ("AVBL", "AVBR", "PVCL", "PVCR", "AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")


def min_pair(resp: np.ndarray) -> float:
    unit = resp / np.maximum(np.linalg.norm(resp, axis=1, keepdims=True), 1e-12)
    n = len(unit)
    return float(min(1.0 - abs(float(unit[i] @ unit[j]))
                     for i in range(n) for j in range(i + 1, n)))


def responses(brain, conn) -> np.ndarray:
    rows = []
    for senses in SITUATIONS:
        drive = np.zeros((1, conn.n))
        for population, level in senses:
            drive[0, list(conn.populations[population])] = level
        rows.append(collect(brain, drive)[0][0])
    return np.stack(rows)


def main() -> None:
    conn = load_worm_with_roles()
    names = [c["name"] for c in json.loads(
        (HERE.parent / "data" / "worm_connectome.json").read_text())["neurons"]]
    brain = build(conn, row_sum=0.5)
    resp = responses(brain, conn)
    centred = resp - resp.mean(axis=0)

    searched = json.loads((HERE.parent / "docs" / "assets" / "readout_search_sensory.json")
                          .read_text())["greedy"]["cells"]
    out = {}
    for label, cells in (("declared", DECLARED), ("searched", tuple(searched))):
        idx = np.array([names.index(c) for c in cells])
        out[label] = {"cells": list(cells),
                      "min_pair_distance": round(min_pair(resp[:, idx] - resp[:, idx].mean(axis=0)), 4)}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nverdict: {'PASS' if out['searched']['min_pair_distance'] > 5 * out['declared']['min_pair_distance'] else 'FAIL'}"
          " (searched must beat declared by 5x on the worst pair)")


if __name__ == "__main__":
    main()
