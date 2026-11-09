"""A3: learn the mapping layer. Fixed connectome as feature extractor; how much rank does it need?

§14 showed the connectome alone never produces locomotion polarity (91% excitatory synapses, no
push-pull). So the honest architecture is: freeze the 302-neuron graph, learn a readout. The only
question that decides the roadmap is **how much readout rank is needed** — because that is where
"growth" would first mean something.

Data: sparse random drives over the 61 sensory cells (the animal's real input space).
Labels: two valence targets computed from the drive itself, so no model output leaks in:
    smooth  v = tanh(3 (a - r))          monotone in attractant-minus-repellent
    gate    v = tanh(4 (a - r - 0.2))    same, but needs a threshold/dead zone
Readout: ridge regression on the settled 302-dim state, truncated to r principal components
(fitted on the training fold only). 5-fold CV.

Pre-registered gate:
  the required rank is the smallest r where held-out Spearman rho reaches within 0.02 of the
  best r. PASS if that rank is <= 4 (a small readout suffices -> no growth needed).
  If the gate target needs r >= 16 while the smooth one needs <= 4, the mapping layer's capacity
  is the thing that scales with task complexity -- that is the first real scaling axis found.

    python probes/probe_mapping_layer.py --samples 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_manifold import build, collect, drives_sensory, load_worm_with_roles  # noqa: E402

RANKS = (1, 2, 4, 8, 16, 32, 64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom else 0.0


def fit_predict(x_train, y_train, x_test, rank: int, ridge: float = 1e-2) -> np.ndarray:
    mean = x_train.mean(axis=0)
    y_mean = float(y_train.mean())                        # intercept: without it predictions keep
    xc = x_train - mean                                   # an offset and MAE/sign are meaningless
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:rank]                                     # rank x features
    feat_train = xc @ basis.T
    feat_test = (x_test - mean) @ basis.T
    gram = feat_train.T @ feat_train + ridge * np.eye(rank)
    coef = np.linalg.solve(gram, feat_train.T @ (y_train - y_mean))
    return feat_test @ coef + y_mean


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
    x, input_dim = drives_sensory(conn, args.samples, args.seed)
    states, sweeps, residual = collect(brain, x)

    attract = np.concatenate([conn.populations[p] for p in ("odour_A", "odour_B", "food")])
    repel = np.array(sorted(conn.populations["pain"]))
    a = x[:, attract].sum(axis=1)
    r = x[:, repel].sum(axis=1)
    targets = {"smooth": np.tanh(3.0 * (a - r)),
               "gate": np.tanh(4.0 * (a - r - 0.2))}

    # baseline: the declared locomotion readout, unfitted
    fwd = np.array(sorted(conn.populations["forward"]))
    rev = np.array(sorted(conn.populations["reverse"]))
    raw_L = states[:, fwd].mean(axis=1) - states[:, rev].mean(axis=1)

    order = np.random.default_rng(args.seed).permutation(len(states))
    folds = np.array_split(order, args.folds)

    result = {"probe": "probe_mapping_layer.py", "samples": args.samples,
              "input_dim": input_dim, "row_sum": args.row_sum,
              "sweeps_max": sweeps, "residual_max": round(residual, 2),
              "baseline_unfitted_rho": {name: round(spearman(raw_L, y), 3)
                                        for name, y in targets.items()},
              "ranks": {}}
    for name, y in targets.items():
        rows = []
        for rank in RANKS:
            rhos, signs, maes = [], [], []
            for test in folds:
                mask = np.ones(len(states), bool)
                mask[test] = False
                pred = fit_predict(states[mask], y[mask], states[test], rank)
                y_test = y[test]
                rhos.append(spearman(pred, y_test))
                strong = np.abs(y_test) >= 0.5            # only labels away from zero
                signs.append(float(np.mean(np.sign(pred[strong]) == np.sign(y_test[strong])))
                             if strong.any() else 0.5)
                maes.append(float(np.mean(np.abs(pred - y_test))))
            rows.append({"rank": rank, "rho_mean": round(float(np.mean(rhos)), 3),
                         "rho_sd": round(float(np.std(rhos)), 3),
                         "mae": round(float(np.mean(maes)), 3),
                         "sign_accuracy_strong": round(float(np.mean(signs)), 3)})
        best = max(row["rho_mean"] for row in rows)
        needed = next((row["rank"] for row in rows if row["rho_mean"] >= best - 0.02), None)
        result["ranks"][name] = {"rows": rows, "best_rho": best, "rank_needed": needed}

    smooth_rank = result["ranks"]["smooth"]["rank_needed"]
    gate_rank = result["ranks"]["gate"]["rank_needed"]
    result["verdict"] = (
        f"小读出够用（smooth 需要 rank {smooth_rank}）；gate 需要 rank {gate_rank}"
        if smooth_rank and smooth_rank <= 4 else "smooth 目标就需要大秩，映射层容量是瓶颈")

    assert len(result["ranks"]["smooth"]["rows"]) == len(RANKS)
    assert all(0.0 <= row["sign_accuracy_strong"] <= 1.0
               for name in targets for row in result["ranks"][name]["rows"])

    out = args.out or (HERE.parent / "docs" / "assets" / "mapping_layer.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"baseline unfitted rho: {result['baseline_unfitted_rho']}  (declared locomotion readout)")
    for name in targets:
        r = result["ranks"][name]
        print(f"\n{name}: best rho={r['best_rho']} rank_needed={r['rank_needed']}")
        for row in r["rows"]:
            print(f"  rank={row['rank']:<3} rho={row['rho_mean']:.3f}±{row['rho_sd']:.3f} "
                  f"mae={row['mae']:.3f} sign(|y|>=.5)={row['sign_accuracy_strong']:.3f}")
    print(f"\n{result['verdict']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
