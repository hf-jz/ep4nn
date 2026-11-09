"""P1: does a two-compartment node buy something a single compartment cannot?

Xu Ninglong's Science 2026 result says the apical dendrite carries the *rule* while the soma carries
the *stimulus*, and the two are integrated nonlinearly. The testable consequence is not "more
capacity" but a specific capability: a context-dependent reversal (XOR between rule and stimulus)
that a single-compartment linear-sum node cannot produce at all, whatever its scalars.

Model (minimal, self-contained; Cadence's NeuronModel has one compartment only):

    single:  v <- (1-a) v + a (-v + W r + d)              r = sigmoid(slope (v - thr))
    two:     a_ap <- (1-tau) a_ap + tau * ctx
             thr_eff = thr * (1 + kappa (2 a_ap - 1))      # rule gates the threshold
             v <- (1-a) v + a (-v + W r + d)              r = sigmoid(slope (v - thr_eff))

W is the measured c302 wiring, row-normalised, signs per cell. The context channel is one designated
sensory cell; the stimulus is another; the label is their XOR. Readout is the same PCA+ridge with
5-fold CV used everywhere else, so the comparison isolates the node, not the readout.

Pre-registered judgement: PASS if two compartments reach >=0.80 held-out XOR accuracy while the
single compartment stays within 0.10 of chance (0.50). If both fail, compartmental gating does not
deliver the capability either, and the electrode moves back to weights (overturn item 3, part two).

    python probes/probe_two_compartment.py --trials 80 --steps 300
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
from probe_mapping_layer import fit_predict  # noqa: E402

SLOPE = 4.0
THR = 0.5
ALPHA = 0.35


def wiring(conn) -> np.ndarray:
    """W[post, pre] = sign * contacts / in-degree, the same operating point as the other probes."""
    indeg = np.bincount(conn.post, minlength=conn.n).astype(float)
    indeg[indeg == 0] = 1.0
    w = np.zeros((conn.n, conn.n))
    np.add.at(w, (conn.post, conn.pre), conn.sign / indeg[conn.post])
    return w


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def settle(w, drive, *, compartments: int, ctx: float, steps: int,
           tau: float = 0.2, kappa: float = 3.0, ctx_cell: int | None = None) -> np.ndarray:
    r = np.zeros(w.shape[0])
    v = np.zeros(w.shape[0])
    apical = 0.0
    if ctx_cell is not None:            # control arm: context arrives as an ordinary additive input,
        drive = drive.copy()            # so the node still has only one compartment
        drive[ctx_cell] += ctx
    for _ in range(steps):
        if compartments == 2:
            apical += tau * (ctx - apical)
            thr = THR * (1.0 + kappa * (2.0 * apical - 1.0))
        else:
            thr = THR
        v = (1 - ALPHA) * v + ALPHA * (-v + w @ r + drive)
        r = sigmoid(SLOPE * (v - thr))
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    w = wiring(conn)
    sensory = [int(c) for c in conn.populations["odour_A"] + conn.populations["odour_B"]]
    stim_cell, ctx_cell = sensory[0], sensory[1]

    rng = np.random.default_rng(args.seed)
    drives, labels, ctxs = [], [], []
    for stim in (0.0, 1.0):
        for ctx in (0.0, 1.0):
            for _ in range(args.trials):
                d = np.zeros(conn.n)
                if stim:
                    d[stim_cell] = 1.0
                drives.append(d)
                ctxs.append(ctx)
                labels.append(float(stim != ctx))          # XOR
    labels = np.array(labels)

    result = {"probe": "probe_two_compartment.py", "trials": args.trials, "steps": args.steps,
              "label": "XOR(stimulus, context)", "arms": {}}
    kappas = (0.3, 0.6, 1.0, 3.0)
    arms = (("1c", 1, None), ("1c_ctx", 1, ctx_cell), ("2c", 2, None))
    for kappa in kappas:
        for label, compartments, inject in arms:
            states = np.stack([settle(w, d, compartments=compartments, ctx=c, steps=args.steps,
                                      kappa=kappa, ctx_cell=inject) for d, c in zip(drives, ctxs)])
            order = rng.permutation(len(states))
            folds = np.array_split(order, args.folds)
            accs = []
            for test in folds:
                mask = np.ones(len(states), bool)
                mask[test] = False
                pred = fit_predict(states[mask], labels[mask], states[test], rank=8)
                accs.append(float(np.mean((pred > 0.5) == (labels[test] > 0.5))))
            acc = float(np.mean(accs))
            centroids = np.stack([states[labels == v].mean(axis=0) for v in (0.0, 1.0)])
            sep = float(np.linalg.norm(centroids[0] - centroids[1])
                        / (np.mean([states[labels == v].std(axis=0).mean() for v in (0.0, 1.0)]) + 1e-9))
            result["arms"][f"{label}_k{kappa}"] = {
                "accuracy": round(acc, 3), "centroid_separation": round(sep, 2),
                "state_rank": int(np.linalg.matrix_rank(states - states.mean(axis=0)))}
            print(f"{label:7} kappa={kappa:<4} accuracy={acc:.3f} "
                  f"sep={sep:.2f} rank={result['arms'][f'{label}_k{kappa}']['state_rank']}",
                  flush=True)

    best = {arm: max(v["accuracy"] for k, v in result["arms"].items() if k.startswith(arm + "_"))
            for arm, _, _ in arms}
    one, control, two = best["1c"], best["1c_ctx"], best["2c"]
    passed = two >= 0.80 and abs(control - 0.50) <= 0.10
    result["verdict"] = ("PASS: 两房室拿到单房室拿不到的能力（规则门控阈值有效）" if passed else
                         "FAIL: 两房室也没解决 → 缺的仍是权重/符号，不是节房室结构")

    assert len(labels) == 4 * args.trials
    assert 0.0 <= one <= 1.0 and 0.0 <= two <= 1.0

    out = args.out or (HERE.parent / "docs" / "assets" / "two_compartment.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"single": one, "two": two, "verdict": result["verdict"]}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
