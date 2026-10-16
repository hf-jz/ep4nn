"""The heterogeneity arm with the library's own calibration, which the first screen switched off.

Per-neuron heterogeneity makes most cells silent, which is the effect the library reports for
per-class gains. A silent readout cannot collide, so the arm has to be calibrated before it is
judged: `calibrate=True` centres the readout's operating point, which is the library's prescription
for exactly this. This runs that arm and then grows in it.

    python probes/probe_hetero_calibrated.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ep4nn.grow import EP4NN, GrowthConfig, load_worm   # noqa: E402

OUT: dict[str, object] = {}


def run(row_sum: float, hetero: float, ticks: int = 400) -> dict:
    ep = EP4NN(load_worm(tone=True),
               GrowthConfig(context="context_tone", row_sum=row_sum, hetero_spread=hetero,
                            calibrate=True))
    codes = ep._situation_codes()
    half = len(codes) // 2
    norms = np.linalg.norm(codes, axis=1)
    coll = 1.0 if norms.min() == 0 else max(
        abs(float(codes[i] @ codes[j]) / (norms[i] * norms[j]))
        for s in (0, half) for i in range(s, s + half) for j in range(i + 1, s + half))
    rng = np.random.default_rng(5)
    for _ in range(ticks):
        c = int(rng.integers(0, 3))
        ep.tick(("odour_A", 1.0), ("context_tone", 0.35 + 0.65 * c / 2.0), 0, c)
    return {"row_sum": row_sum, "hetero": hetero, "calibrated": True,
            "readout_spread": round(ep.readout_spread, 5),
            "pair_motion": round(ep.pair_motion, 5), "state_motion": round(ep.state_motion, 6),
            "collision_before": round(float(coll), 6),
            "mean_activation": round(float(np.mean(codes)), 5),
            "kept": ep.growths_kept, "rolled_back": ep.growths_failed,
            "refused": len([e for e in ep.log if e["event"] == "no_response"])}


def main() -> None:
    rows = [run(rs, h) for h in (0.0, 0.5) for rs in (0.5, 1.0, 2.0)]
    OUT["rows"] = rows
    for r in rows:
        print(f"h={r['hetero']} row_sum={r['row_sum']}: spread {r['readout_spread']:.5f} "
              f"state {r['state_motion']:.6f} collision {r['collision_before']:.6f} "
              f"act {r['mean_activation']:.5f} -> kept {r['kept']} rolled_back {r['rolled_back']} "
              f"refused {r['refused']}")
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "hetero_calibrated.json"
    dump.write_text(json.dumps(OUT, indent=1) + "\n")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
