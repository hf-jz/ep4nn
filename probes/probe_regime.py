"""Which regime actually lets an added synapse be seen?

The first screen answered a question I had wrong: the answer cells were *not* saturated in the tone
substrate, they were below threshold, where the rectified sigmoid has zero slope. A cell whose
response gain is zero cannot show the effect of an added input, however the input is wired. So the
quantity that decides whether a structural repair can be visible is the answer's local response gain,
and the regime knobs that set it are the operating point (row sum), the input scale, and per-neuron
heterogeneity. This screens those three, reports the gain, and grows only where the gain is not zero.

    python probes/probe_regime.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ep4nn.grow import EP4NN, GrowthConfig, load_worm, operating_point   # noqa: E402
from ep4nn.hetero import HeteroRateModel                                # noqa: E402
from cadence.brain import Brain                                         # noqa: E402

OUT: dict[str, object] = {}
LEVELS = np.linspace(0.2, 1.0, 5)
ROW_SUMS = (0.5, 1.0, 2.0, 4.0)
DRIVES = (1.0, 3.0, 10.0)
HETERO = (0.0, 0.5)


def measure(conn, readout, row_sum, drive, hetero, levels=LEVELS) -> dict:
    eff = operating_point(conn, row_sum)
    model = (HeteroRateModel(n=conn.n, spread=hetero, seed=3) if hetero > 0
             else __import__("cadence.learning", fromlist=["learning_neuron_model"]).learning_neuron_model())
    brain = Brain(conn, model, backend="cpu", efficacy=eff)
    codes, vs = [], []
    for lv in levels:
        d = np.zeros((1, conn.n))
        d[0, list(conn.populations["odour_A"])] = 1.0 * drive
        d[0, list(conn.populations["context_tone"])] = float(lv) * drive
        eq = brain.equilibrate(d, budget=512, chunk=32, tolerance=1e-3)
        codes.append(np.asarray(eq.state.activation).ravel()[readout])
        vs.append(np.asarray(eq.state.v).ravel())            # the whole state: the slope is per neuron
    codes, v_full = np.stack(codes), np.stack(vs)
    vs = v_full[:, readout]
    norms = np.linalg.norm(codes, axis=1)
    coll = 1.0 if norms.min() == 0 else max(
        abs(float(codes[i] @ codes[j]) / (norms[i] * norms[j])) for i in range(len(codes)) for j in range(i + 1, len(codes)))
    slope = np.stack([np.asarray(model.slope_at(v_full[i]))[readout]
                      for i in range(len(v_full))]).ravel()
    v_flat = vs.ravel()
    return {"row_sum": row_sum, "drive": drive, "hetero": hetero,
            "collision": round(float(coll), 6),
            "mean_v": round(float(np.mean(v_flat)), 4),
            "mean_slope": round(float(np.mean(slope)), 5),
            "mean_activation": round(float(np.mean(codes)), 5),
            "max_activation": round(float(np.max(codes)), 5),
            "responsive_cells": round(float(np.mean((slope > 0.05) & (codes.ravel() < 0.9))), 4)}


def grow_here(row_sum: float, drive: float, hetero: float, init_scale: float, ticks: int = 400) -> dict:
    ep = EP4NN(load_worm(tone=True),
               GrowthConfig(context="context_tone", row_sum=row_sum, drive_scale=drive,
                            hetero_spread=hetero, init_scale=init_scale, calibrate=False))
    rng = np.random.default_rng(5)
    for _ in range(ticks):
        c = int(rng.integers(0, 3))
        ep.tick(("odour_A", 1.0), ("context_tone", 0.35 + 0.65 * c / 2.0), 0, c)
    return {"row_sum": row_sum, "drive": drive, "hetero": hetero, "init_scale": init_scale,
            "kept": ep.growths_kept, "rolled_back": ep.growths_failed,
            "refused": len([e for e in ep.log if e["event"] == "no_response"]),
            "neurons": ep.neurons}


def main() -> None:
    conn = load_worm(tone=True)
    readout = sorted(set(conn.populations["forward"]) | set(conn.populations["reverse"]))
    rows = [measure(conn, readout, rs, d, h) for h in HETERO for rs in ROW_SUMS for d in DRIVES]
    OUT["screen"] = rows
    print("regime screen: can the answer show an added input?")
    print(" row_sum drive hetero  collision  mean_v  mean_slope  mean_act  resp_cells")
    for r in rows:
        print(f"  {r['row_sum']:>6} {r['drive']:>6} {r['hetero']:>6}  {r['collision']:>9.6f} "
              f"{r['mean_v']:>7.3f} {r['mean_slope']:>10.5f} {r['mean_activation']:>9.5f} "
              f"{r['responsive_cells']:>10.4f}")
    good = [r for r in rows if r["mean_slope"] > 0.05 and r["collision"] > 0.85
            and r["mean_activation"] > 1e-4]
    OUT["responsive"] = good
    print(f"\n{len(good)} of {len(rows)} regimes are responsive, collide, and are not silent")
    runs = []
    for r in good[:6]:
        for init in (3.0, 10.0):
            run = grow_here(r["row_sum"], r["drive"], r["hetero"], init)
            runs.append(run)
            print(f"  growth row_sum={run['row_sum']} drive={run['drive']} hetero={run['hetero']} "
                  f"init={run['init_scale']}: kept={run['kept']} rolled_back={run['rolled_back']} "
                  f"refused={run['refused']} neurons={run['neurons']}")
    OUT["growth"] = runs
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "regime.json"
    dump.write_text(json.dumps(OUT, indent=1) + "\n")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
