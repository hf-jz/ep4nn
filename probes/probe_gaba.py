"""Measured sign layer, take two: signs are per-cell and biologically correct — what about strength?

`probes/analyse_signs.py` shows the c302 sign layer is not scattered noise: 27 cells are
all-inhibitory and 244 all-excitatory, with no mixed cell, and the 27 match the canonical
GABAergic set of the hermaphrodite (AVL, DVB, RIS, RIAL/RIAR, DD/VD, RME, DVC). So §14's
"91 percent excitatory" was a count, not a defect: the polarity failure is not a missing sign.

What is missing is *strength*: row normalisation gives every postsynaptic cell the same total
input, so a GABAergic neuron's inhibition is diluted by its in-degree and scaled identically to
every excitatory edge.

This probe scales the efficacy of GABAergic efferents by gamma and asks whether the locomotion
command ever reverses. Control: the same gamma applied to 27 random cells, to show the effect is
not "more inhibition anywhere".

Pre-registered judgement: PASS (sign layer works, strength was the defect) if gamma in {2,5,10,20}
produces at least one sign reversal that tracks the expected valence (rho >= 0.8). FAIL otherwise.

    python probes/probe_gaba.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_locomotion import CONDITIONS, spearman  # noqa: E402
from probe_manifold import collect, load_worm_with_roles  # noqa: E402

from cadence.brain import Brain  # noqa: E402
from cadence.learning import learning_neuron_model  # noqa: E402

GAMMAS = (1.0, 2.0, 5.0, 10.0, 20.0)


def gaba_cells(conn) -> np.ndarray:
    """Cells whose outgoing *chemical* edges are all inhibitory — the derived GABAergic set.

    Taken from the raw file, not from the Connectome: the Connectome also carries gap junctions
    with sign +1, which would make every cell look mixed.
    """
    raw = json.loads((HERE.parent / "data" / "worm_connectome.json").read_text())
    names = [c["name"] for c in raw["neurons"]]
    out: dict[int, list[float]] = {}
    for pre, _post, _count, sign in ((int(r[0]), int(r[1]), float(r[2]), float(r[3]))
                                     for r in raw["chemical"]):
        out.setdefault(pre, []).append(sign)
    return np.array([cell for cell, signs in out.items() if signs and all(s < 0 for s in signs)])


def build_with_gain(conn, row_sum: float, boost: np.ndarray, gamma: float):
    indeg = np.bincount(conn.post, minlength=conn.n).astype(float)
    indeg[indeg == 0] = 1.0
    eff = conn.sign * (row_sum / indeg[conn.post])
    mask = np.isin(conn.pre, boost)
    eff = np.where(mask, eff * gamma, eff)
    return Brain(conn, learning_neuron_model(), backend="cpu", efficacy=eff)


def run(conn, brain) -> tuple[list[float], list[float]]:
    forward = np.array(sorted(conn.populations["forward"]))
    reverse = np.array(sorted(conn.populations["reverse"]))
    Ls, bias = [], []
    for _, senses, expected in CONDITIONS:
        drive = np.zeros((1, conn.n))
        for population, level in senses:
            drive[0, list(conn.populations[population])] = level
        act = collect(brain, drive)[0][0]
        Ls.append(float(act[forward].mean() - act[reverse].mean()))
        bias.append(expected)
    return Ls, bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    gaba = gaba_cells(conn)
    rng = np.random.default_rng(args.seed)
    everyone = np.arange(conn.n)
    random_set = rng.choice(np.setdiff1d(everyone, gaba), size=len(gaba), replace=False)

    result = {"probe": "probe_gaba.py", "gaba_cells": [int(c) for c in gaba],
              "gaba_count": int(len(gaba)), "row_sum": args.row_sum, "arms": {}}

    for label, boost in (("gaba", gaba), ("random", random_set)):
        rows = []
        for gamma in GAMMAS:
            brain = build_with_gain(conn, args.row_sum, boost, gamma)
            Ls, bias = run(conn, brain)
            rho = spearman(Ls, bias)
            signs_ok = all((L > 0) == (b > 0) for L, b in zip(Ls, bias) if b != 0)
            rows.append({"gamma": gamma, "rho": round(rho, 3), "signs_agree": signs_ok,
                         "L_min": round(min(Ls), 6), "L_max": round(max(Ls), 6)})
            print(f"{label:7} gamma={gamma:<5} rho={rho:+.3f} signs={signs_ok} "
                  f"L=[{min(Ls):+.5f}, {max(Ls):+.5f}]", flush=True)
        result["arms"][label] = rows

    best = max((r["rho"] for r in result["arms"]["gaba"] if r["gamma"] > 1), default=-1)
    result["verdict"] = ("PASS: 符号层没问题，缺的是强度（GABA 增益下极性出现）" if best >= 0.8
                         else "FAIL: 即使把 GABA 强度放到 20× 也不出现极性 → 缺的是实测权重矩阵")

    assert (len(gaba) == 27), len(gaba)
    out = args.out or (HERE.parent / "docs" / "assets" / "gaba.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"gaba_count": int(len(gaba)), "best_gaba_rho": best,
                      "verdict": result["verdict"]}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
