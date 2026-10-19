"""Fly scale: the same substrate contract and the same metric, on a 139k-neuron connectome.

`/Users/sheng/bionic_brain/flybrain` carries FlyWire FAFB v783 as a binary dump written by
`scripts/build_connectome.py`: a `<II` header (neurons, edges), then `<IIf` per edge, then `<BH` per
neuron (region, group). Four regions: 0 sensory, 1 central, 2 drives, 3 motor. That is exactly what the
port ABI needs, so nothing here is species-specific — it is the worm probe at 460 times the size.

    python probes/probe_fly.py
"""

from __future__ import annotations

import gzip
import json
import resource
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ep4nn.substrate import PopulationSpec, Substrate, worm_substrate   # noqa: E402
from cadence import Connectome                                        # noqa: E402

DATA = Path("/Users/sheng/bionic_brain/flybrain/data")
BIN = DATA / "connectome.bin.gz"
REGIONS = {0: "sensory", 1: "central", 2: "drives", 3: "motor"}


def transmitter_signs(n_neurons: int) -> np.ndarray | None:
    """Sign per neuron from FlyWire's nt_type: GABAergic cells inhibit. None if unavailable."""
    import csv

    try:                                    # neurons.csv.gz is the canonical order *and* carries nt_type
        rows = list(csv.DictReader(gzip.open(DATA / "neurons.csv.gz", "rt")))
        order = [row["root_id"] for row in rows]
        nt = {row["root_id"]: (row.get("nt_type") or "").strip().upper() for row in rows}
    except (FileNotFoundError, KeyError) as exc:
        print(f" no transmitter table ({exc}); signs default to +1")
        return None
    if len(order) != n_neurons:
        print(f" neuron order is {len(order)}, dump is {n_neurons}; signs default to +1")
        return None
    signs = np.ones(n_neurons, dtype=np.float32)
    signs[[i for i, rid in enumerate(order) if nt.get(rid) == "GABA"]] = -1.0
    print(f" transmitter: {(signs < 0).sum():,} inhibitory of {n_neurons:,}")
    return signs


def load_fly(inhibitory: bool = False) -> Connectome:
    raw = gzip.decompress(BIN.read_bytes())
    n_neurons, n_edges = struct.unpack("<II", raw[:8])
    edges = np.frombuffer(raw, dtype=np.dtype([("pre", "<u4"), ("post", "<u4"), ("w", "<f4")]),
                          count=n_edges, offset=8)
    cells = np.frombuffer(raw, dtype=np.dtype([("region", "u1"), ("group", "<u2")]),
                          count=n_neurons, offset=8 + 12 * n_edges)
    region = cells["region"]
    pops = {name: np.flatnonzero(region == code) for code, name in REGIONS.items()}
    cell_sign = transmitter_signs(n_neurons) if inhibitory else None
    edge_sign = (np.ones(n_edges, dtype=np.float32) if cell_sign is None
                 else cell_sign[edges["post"]])       # a GABAergic postsynaptic cell inhibits
    return Connectome.from_synapses(n_neurons, pre=edges["pre"], post=edges["post"],
                                    count=np.ones(n_edges, dtype=np.int32),
                                    sign=edge_sign,
                                    populations={**pops, "group": cells["group"]},
                                    label=f"fly {n_neurons}")


def main() -> None:
    out = {}
    print(f"loading {BIN.name} ...")
    conn = load_fly()
    fly = Substrate(conn, PopulationSpec(sensory="sensory", association="central", motor="motor"),
                    label=f"fly FAFB {conn.n}")
    print(f"{fly.label}: {conn.n:,} neurons, {conn.synapses:,} synapses, "
          f"populations {[(k, len(np.ravel(v))) for k, v in conn.populations.items() if k != 'group']}")
    out["fly"] = {"neurons": conn.n, "synapses": conn.synapses}
    fly_inh = Substrate(load_fly(inhibitory=True),
                        PopulationSpec(sensory="sensory", association="central", motor="motor"),
                        label=f"fly FAFB, inhibition")
    for label, sub in (("worm 302", worm_substrate()), ("fly FAFB, all excitatory", fly),
                       ("fly FAFB, inhibition", fly_inh)):
        got = sub.measure()
        out[label] = got
        print(f" {label:<10} {got}")
    out["peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 1)
    print(f" peak RSS {out['peak_mb']} MB")
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "fly.json"
    dump.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
