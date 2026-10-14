"""Run receipts: nominal versus executed, and what the premise actually measured.

Adopted from the open fly-connectome projects, whose descriptions and executed
graphs differ widely (a browser demo described as a 166,000-neuron connectome runs
an 850-neuron stand-in; a sub-network study is described by its dataset size). A
receipt is the cheapest fix: six fields, written by the run that produced them, so
two results can be compared without reading anyone's prose.

Fields
    dataset     source, version, hash of the graph actually loaded
    run         nominal counts (in the file) versus executed counts (in this run)
    dynamics    model name, its declared parameters, what is absent (delay, adaptation)
    loop        whether the loop is closed, and which parts are supplied by hand
    premise     the prerequisite measurements: readout spread, collisions, warnings
    cost        per sweep, sweeps, rebuild, memory, and the operation counts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIELDS = ("dataset", "run", "dynamics", "loop", "premise", "cost")


def build(*, dataset: dict[str, Any], nominal: dict[str, int], executed: dict[str, int],
          dynamics: dict[str, Any], loop: dict[str, Any], premise: dict[str, Any],
          cost: dict[str, Any]) -> dict[str, Any]:
    """One receipt. Every field is explicit, including the parts that are supplied."""
    missing = [k for k, v in dict(nominal=nominal, executed=executed, dynamics=dynamics,
                                  loop=loop, premise=premise, cost=cost).items() if not v]
    if missing:
        raise ValueError(f"a receipt field cannot be empty: {missing}")
    if not dataset.get("source") or not dataset.get("sha256"):
        raise ValueError("a dataset needs a source and a hash")
    return {
        "format": "ep4nn-receipt/1",
        "dataset": dataset,
        "run": {"nominal": nominal, "executed": executed,
                "scale_ratio": {k: (round(executed[k] / nominal[k], 4) if nominal.get(k) else None)
                                for k in executed}},
        "dynamics": dynamics,
        "loop": loop,
        "premise": premise,
        "cost": cost,
    }


def write(receipt: dict[str, Any], path: str | Path) -> Path:
    """Write the receipt next to its run, atomically, sorted for a stable diff."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)
    return p


def _self_check() -> None:
    r = build(
        dataset={"source": "openworm/c302 A_Full", "sha256": "2a7b117f"},
        nominal={"neurons": 302, "synapses": 3137}, executed={"neurons": 302, "synapses": 3137},
        dynamics={"model": "rate", "gain": 1.0, "absent": ["delay", "adaptation"]},
        loop={"closed": False, "supplied": ["senses", "outcome"]},
        premise={"readout_spread": 0.019, "collision_max": 1.0},
        cost={"per_sweep_us": 98.4, "sweeps": 32},
    )
    assert r["format"] == "ep4nn-receipt/1" and set(r) >= set(FIELDS) | {"format"}
    assert r["run"]["scale_ratio"]["neurons"] == 1.0
    for bad in (dict(dataset={"source": "x"}, nominal={}, executed={}, dynamics={}, loop={},
                     premise={}, cost={}),):
        try:
            build(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError("an empty field or an unhashed dataset must be refused")
    print("receipt self-check ok")


if __name__ == "__main__":
    _self_check()
