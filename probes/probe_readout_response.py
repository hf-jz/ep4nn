"""Phase 0 probe: does anything reach the readout at all?

The blocker ep4nn hit was `readout spread = 0.019`: every situation produced the
same answer, so capacity growth was refused (correctly). This scans the declared
operating point — row sum, and a gain on one cell class — for a configuration in
which the declared readout (forward AVB/PVC + reverse AVA/AVD/AVE, 10 cells)
actually moves, and measures the collision it leaves behind.

Two numbers decide, and nothing else:
  spread      mean over readout cells of (max - min) across situations
  collision   largest |cosine| between two situations that share the present
              input and differ only in context (must differ, so this must fall)

Roles come from the connectome file (openworm/c302 cell classes via
ConnectomeToolbox), which is the only "cell class" the wiring carries.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.brain import Brain
from cadence.learning import learning_neuron_model

DATA = Path(__file__).resolve().parents[1] / "data" / "worm_connectome.json"
OUT: dict[str, object] = {}


def load_worm_with_roles() -> Connectome:
    raw = json.loads(DATA.read_text())
    names = [c["name"] for c in raw["neurons"]]
    roles = [c.get("role", "unknown") for c in raw["neurons"]]
    index = {name: i for i, name in enumerate(names)}
    pre, post, sign = [], [], []
    for a, b, _, s in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
        if a != b:
            pre.append(a); post.append(b); sign.append(s)
    for a, b, _ in ((int(r[0]), int(r[1]), float(r[2])) for r in raw["gap"]):
        if a != b:
            for x, y in ((a, b), (b, a)):
                pre.append(x); post.append(y); sign.append(1.0)
    populations: dict[str, tuple[int, ...]] = {
        "odour_A": tuple(index[n] for n in ("AWAL", "AWAR")),
        "odour_B": tuple(index[n] for n in ("AWCL", "AWCR")),
        "food": tuple(index[n] for n in ("CEPDL", "CEPDR", "CEPVL", "CEPVR",
                                         "ADEL", "ADER", "PDEL", "PDER")),
        "pain": tuple(index[n] for n in ("ASHL", "ASHR")),
        "forward": tuple(index[n] for n in ("AVBL", "AVBR", "PVCL", "PVCR")),
        "reverse": tuple(index[n] for n in ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")),
    }
    for role in sorted(set(roles)):
        populations[f"class_{role}"] = tuple(i for i, r in enumerate(roles) if r == role)
    return Connectome.from_synapses(len(names), pre=pre, post=post, sign=sign,
                                    populations=populations, label="celegans:c302")


SITUATIONS = [
    (("odour_A", 1.0),),
    (("odour_B", 1.0),),
    (("food", 1.0),),
    (("pain", 1.0),),
    (("odour_A", 1.0), ("pain", 0.35)),
    (("odour_A", 1.0), ("pain", 0.70)),
    (("odour_A", 1.0), ("pain", 1.00)),
    (("odour_B", 1.0), ("pain", 0.35)),
    (("odour_B", 1.0), ("pain", 1.00)),
]
# pairs (i, j) that share the present input and differ only in context
PAIRS = [(4, 5), (4, 6), (5, 6), (7, 8)]
READOUT = ("forward", "reverse")


def run(conn: Connectome, row_sum: float, class_scale: float, gain_class: str | None,
        tolerance: float = 1e-3, budget: int = 512) -> dict:
    indeg = np.bincount(conn.post, minlength=conn.n).astype(float)
    indeg[indeg == 0] = 1.0
    efficacy = conn.sign * (row_sum / indeg[conn.post])
    log_gain = np.zeros(conn.n)
    if gain_class is not None and class_scale != 1.0:
        log_gain[list(conn.populations[gain_class])] = float(np.log(class_scale))
    brain = Brain(conn, learning_neuron_model(), backend="cpu", efficacy=efficacy, log_gain=log_gain)
    readout = np.array(sorted(set(conn.populations["forward"]) | set(conn.populations["reverse"])))

    codes, infos = [], []
    for senses in SITUATIONS:
        drive = np.zeros((1, conn.n))
        for population, level in senses:
            drive[0, list(conn.populations[population])] = level
        eq = brain.equilibrate(drive, budget=budget, chunk=32, tolerance=tolerance)
        codes.append(np.asarray(eq.state.activation).ravel()[readout])
        infos.append((int(eq.steps), float(np.max(eq.residual))))
    codes_a = np.stack(codes)
    spread = float(np.mean(codes_a.max(axis=0) - codes_a.min(axis=0)))
    norms = np.linalg.norm(codes_a, axis=1)
    coll = []
    for i, j in PAIRS:
        if norms[i] > 0 and norms[j] > 0:
            coll.append(abs(float(codes_a[i] @ codes_a[j]) / (norms[i] * norms[j])))
    return {
        "row_sum": row_sum, "gain_class": gain_class, "class_scale": class_scale,
        "spread": round(spread, 4),
        "collision_max": round(max(coll), 4) if coll else None,
        "collision_mean": round(float(np.mean(coll)), 4) if coll else None,
        "sweeps_max": int(max(s for s, _ in infos)),
        "residual_max": round(max(r for _, r in infos), 2),
        "answer_norm": [round(float(x), 3) for x in norms],
    }


def main() -> None:
    conn = load_worm_with_roles()
    rows = []
    for row_sum, gain_class, scale in product((0.1, 0.2, 0.5, 1.0, 2.0),
                                              (None, "class_interneuron", "class_command"),
                                              (1.0, 2.0, 5.0, 10.0)):
        if gain_class is None and scale != 1.0:
            continue
        rows.append(run(conn, row_sum, scale, gain_class))
    rows.sort(key=lambda r: (-r["spread"] if r["spread"] == r["spread"] else 0))
    OUT["top"] = rows[:12]
    OUT["bottom"] = rows[-4:]
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "prereq_scan.json"
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(json.dumps({"probe": "probe_readout_response.py", "rows": rows,
                                "situations": SITUATIONS, "pairs": PAIRS}, indent=1))
    print(f"wrote {dump} ({len(rows)} rows)")
    print(json.dumps(OUT, indent=2))
    print("\nrow_sum | gain_class        | scale | spread | coll_max | sweeps | res")
    for r in rows[:16]:
        print(f"{r['row_sum']:>7} | {str(r['gain_class']):>17} | {r['class_scale']:>5} | "
              f"{r['spread']:>6} | {r['collision_max']!s:>8} | {r['sweeps_max']:>6} | {r['residual_max']}")


if __name__ == "__main__":
    main()
