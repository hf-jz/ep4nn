"""A0: is the 302-neuron state rich, or has it collapsed? (manifold diagnostic)

The growth ledger refuses 1200/1200 because the declared readout moves 0.019 while the
answer does not move. Two very different causes, and the difference decides the whole
roadmap:

  collapse     the state itself barely moves (effective rank ~1) -> adding neurons adds
               copies of the same collapsed state; the fix is a decorrelation/anti-collapse
               term and a richer node model, not more nodes.
  leakage      the state does move (rank high) but the declared readout (forward AVB/PVC +
               reverse AVA/AVD/AVE) does not carry that movement -> the bottleneck is the
               readout/mapping, not capacity and not the node model.

Measured, on the same connectome / neuron model / operating point as
``probe_readout_response.py``:

  participation_ratio  (sum lambda)^2 / sum lambda^2 over the centred 302-dim state sample
  k90 / k99            dimensions needed for 90% / 99% of the state variance
  pc1_share            share of variance in the first component
  state_rms            mean over samples of ||activation - mean|| / sqrt(n)
  readout_rms          the same restricted to the 10 declared readout cells
  leakage              readout_rms / state_rms  (how much state movement reaches the readout)
  readout_rank_share   k90 of the readout-only sample, divided by k90 of the full state

Decision rule (fixed before running, do not tune after):
  PR < 3 or k90 <= 2                      -> collapse. Do not grow. Fix node model / add
                                             an anti-collapse term.
  leakage < 0.20 and PR >= 3               -> mapping. Do not grow. Re-pick the readout.
  leakage >= 0.20 and PR >= 3              -> capacity story stays open; growth may be
                                             testable with the manifold-distance gate.

    python probes/probe_manifold.py --samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_readout_response import READOUT, load_worm_with_roles  # noqa: E402

from cadence.brain import Brain  # noqa: E402  (low-level graph; cadence.Brain is the compose wrapper)
from cadence.learning import learning_neuron_model  # noqa: E402

SENSORY = ("odour_A", "odour_B", "food", "pain")


def build(conn, row_sum: float = 0.5, class_scale: float = 1.0, gain_class: str | None = None,
          slope: float | None = None, leak: float | None = None,
          bias_jitter: float = 0.0, gain_jitter: float = 0.0, seed: int = 0,
          efficacy: str = "row", adaptation=None):
    from dataclasses import replace

    indeg = np.bincount(conn.post, minlength=conn.n).astype(float)
    indeg[indeg == 0] = 1.0
    if efficacy == "row":
        # every post neuron receives the same total input -- erases the topology's asymmetry
        eff = conn.sign * (row_sum / indeg[conn.post])
    elif efficacy == "raw":
        # keep the raw contact counts; scale only so the mean weight matches the row mode
        eff = conn.sign * conn.count * (row_sum / float(conn.count.mean()))
    elif efficacy == "sqrt":
        eff = conn.sign * conn.count * row_sum / np.sqrt(indeg[conn.post])
    else:
        raise ValueError(f"unknown efficacy mode {efficacy!r}")
    log_gain = np.zeros(conn.n)
    if gain_class is not None and class_scale != 1.0:
        log_gain[list(conn.populations[gain_class])] = float(np.log(class_scale))
    if bias_jitter or gain_jitter:
        # A1b: break the symmetry that pins every neuron to the same mode. Per-neuron bias and
        # gain are arrays Brain already accepts, so this needs no new neuron class.
        rng = np.random.default_rng(seed + 7919)
        bias = rng.normal(0.0, bias_jitter, conn.n) if bias_jitter else np.zeros(conn.n)
        log_gain = log_gain + (rng.normal(0.0, gain_jitter, conn.n) if gain_jitter else 0.0)
    else:
        bias = None
    model = learning_neuron_model()
    if slope is not None or leak is not None or adaptation is not None:
        # A1: node expressivity, not node count. slope = sigmoid steepness (gain of the
        # transfer), leak = the 0..1 memory of the previous potential, adaptation = the slow
        # variable (SPN-style persistent state).
        model = replace(model, slope=model.slope if slope is None else slope,
                        leak=model.leak if leak is None else leak,
                        adaptation=adaptation if adaptation is not None else model.adaptation)
    return Brain(conn, model, backend="cpu", efficacy=eff, log_gain=log_gain, bias=bias)


def drives(conn, samples: int, seed: int) -> np.ndarray:
    """Random levels on the four sensory populations, plus the canonical one-hot corners."""
    rng = np.random.default_rng(seed)
    rows = []
    for population in SENSORY:                       # corners
        row = np.zeros(conn.n)
        row[list(conn.populations[population])] = 1.0
        rows.append(row)
    for _ in range(samples):                         # interior: random subset, random levels
        row = np.zeros(conn.n)
        for population in SENSORY:
            if rng.random() < 0.6:
                row[list(conn.populations[population])] = float(rng.uniform(0.05, 1.0))
        rows.append(row)
    return np.stack(rows)


def sensory_cells(conn) -> list[int]:
    """Every cell the connectome labels sensory -- the honest input space of this animal."""
    roles = [c.get("role", "unknown") for c in json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "worm_connectome.json").read_text())["neurons"]]
    return [i for i, role in enumerate(roles) if role == "sensory"]


def drives_sensory(conn, samples: int, seed: int, density: float = 0.25) -> tuple[np.ndarray, int]:
    """High-dimensional input: sparse random patterns over all sensory cells.

    This removes the confound of the 4-population drive, whose own dimensionality caps the
    state's participation ratio at 4.
    """
    rng = np.random.default_rng(seed)
    cells = sensory_cells(conn)
    rows = []
    for _ in range(samples):
        row = np.zeros(conn.n)
        active = rng.random(len(cells)) < density
        if not active.any():
            active[rng.integers(len(cells))] = True
        row[np.array(cells)[active]] = rng.uniform(0.05, 1.0, int(active.sum()))
        rows.append(row)
    return np.stack(rows), len(cells)


def spectrum(x: np.ndarray) -> dict:
    """x: samples x neurons, already centred."""
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = float(var.sum())
    if total <= 0:
        return {"participation_ratio": 1.0, "k90": 0, "k99": 0, "pc1_share": 1.0, "rank": 0}
    share = var / total
    k90 = int(np.searchsorted(np.cumsum(share), 0.90) + 1)
    k99 = int(np.searchsorted(np.cumsum(share), 0.99) + 1)
    pr = float(total ** 2 / float((var ** 2).sum()))
    return {"participation_ratio": round(pr, 3), "k90": k90, "k99": k99,
            "pc1_share": round(float(share[0]), 3), "rank": int(np.linalg.matrix_rank(x))}


def collect(brain, x: np.ndarray, tolerance: float = 1e-3, budget: int = 512):
    """Settle every drive and return (states, sweeps_max, residual_max)."""
    if not len(x):
        raise ValueError("need at least one drive")
    states, sweeps = [], []
    residual = np.zeros(1)
    for drive in x:
        eq = brain.equilibrate(drive.reshape(1, -1), budget=budget, chunk=32, tolerance=tolerance)
        states.append(np.asarray(eq.state.activation).ravel())
        sweeps.append(int(eq.steps))
        residual = eq.residual
    return np.stack(states), max(sweeps), float(np.max(residual))


def measure(brain, x: np.ndarray, readout: np.ndarray, tolerance: float, budget: int) -> dict:
    state, sweeps_max, residual_max = collect(brain, x, tolerance, budget)
    centred = state - state.mean(axis=0)
    full = spectrum(centred)
    ro = centred[:, readout]
    ro_stats = spectrum(ro)

    state_rms = float(np.mean(np.linalg.norm(centred, axis=1) / np.sqrt(state.shape[1])))
    ro_rms = float(np.mean(np.linalg.norm(ro, axis=1) / np.sqrt(ro.shape[1])))
    return {"n_samples": int(state.shape[0]), "n_neurons": int(state.shape[1]),
            "sweeps_max": int(sweeps_max), "residual_max": round(residual_max, 2),
            "full": full, "readout": ro_stats,
            "state_rms": round(state_rms, 6), "readout_rms": round(ro_rms, 6),
            "leakage": round(ro_rms / state_rms, 4) if state_rms > 0 else None,
            "readout_rank_share": round(ro_stats["k90"] / full["k90"], 3) if full["k90"] else None,
            "readout_share_of_var": round(float(ro.var() / centred.var()), 4)}


def verdict(m: dict) -> str:
    pr = m["full"]["participation_ratio"]
    if pr < 3 or m["full"]["k90"] <= 2:
        return "collapse: 状态塌了 → 不增长；给节点加表达力 / 加去相关项"
    if m["leakage"] is not None and m["leakage"] < 0.20:
        return "mapping: 状态会动但没到声明的读出 → 不增长；重选读出"
    return "capacity-open: 状态富且读出能吃进去 → 增长可用流形外距离来判"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--out", type=Path, help="default docs/assets/manifold.json")
    parser.add_argument("--slope", type=float, help="A1: sigmoid steepness override")
    parser.add_argument("--leak", type=float, help="A1: potential memory 0..1 override")
    parser.add_argument("--bias-jitter", type=float, default=0.0, help="A1b: per-neuron bias sigma")
    parser.add_argument("--gain-jitter", type=float, default=0.0, help="A1b: per-neuron log-gain sigma")
    parser.add_argument("--input-mode", choices=("pops", "sensory"), default="pops",
                        help="pops = the 4 declared sensory populations; sensory = all sensory cells")
    args = parser.parse_args()

    conn = load_worm_with_roles()
    brain = build(conn, row_sum=args.row_sum, slope=args.slope, leak=args.leak,
                  bias_jitter=args.bias_jitter, gain_jitter=args.gain_jitter, seed=args.seed)
    readout = np.array(sorted(set(conn.populations["forward"]) | set(conn.populations["reverse"])))
    if args.input_mode == "sensory":
        x, input_dim = drives_sensory(conn, args.samples, args.seed)
    else:
        x, input_dim = drives(conn, args.samples, args.seed), 4
    drive_spectrum = spectrum((x - x.mean(axis=0))[:, (x != 0).any(axis=0)])
    result = {"probe": "probe_manifold.py", "row_sum": args.row_sum, "seed": args.seed,
              "slope": args.slope, "leak": args.leak,
              "bias_jitter": args.bias_jitter, "gain_jitter": args.gain_jitter,
              "input_mode": args.input_mode, "input_dim": input_dim,
              "input_spectrum": drive_spectrum,
              "readout_cells": len(readout)}
    result.update(measure(brain, x, readout, args.tolerance, args.budget))
    result["verdict"] = verdict(result)

    assert result["n_neurons"] == conn.n == 302, result["n_neurons"]
    assert np.isfinite(result["state_rms"]) and result["full"]["k90"] >= 1

    out = args.out or (Path(__file__).resolve().parents[1] / "docs" / "assets" / "manifold.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("row_sum", "leakage", "verdict")},
                     ensure_ascii=False))
    print(json.dumps(result["full"]))
    print(json.dumps(result["readout"]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
