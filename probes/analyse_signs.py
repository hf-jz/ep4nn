"""Are the c302 signs per-cell (transmitter identity) or scattered edge noise?

If signs are per-cell, every outgoing chemical edge of a GABAergic neuron is -1 and every other
neuron's edges are +1. If they are scattered, a single neuron has a mix and the sign layer cannot
be repaired by a cell-level classification — only by measured per-synapse weights, which do not
exist for this animal.

Prints the distribution of per-cell inhibitory fractions and the implied GABAergic set, so the
locomotion probe can force a clean per-cell sign layer and test whether polarity appears.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sheng/math/ep4nn")
raw = json.loads((ROOT / "data" / "worm_connectome.json").read_text())
names = [c["name"] for c in raw["neurons"]]

out_edges: dict[int, list[float]] = defaultdict(list)
for pre, post, count, sign in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
    out_edges[pre].append(sign)

fractions = []
for cell, signs in out_edges.items():
    arr = np.array(signs)
    fractions.append((float(np.mean(arr < 0)), names[cell], len(arr)))

fractions.sort(reverse=True)
pure_inh = [f for f in fractions if f[0] == 1.0]
mostly_inh = [f for f in fractions if 0.5 <= f[0] < 1.0]
mixed = [f for f in fractions if 0 < f[0] < 0.5]
pure_exc = [f for f in fractions if f[0] == 0.0]

print(f"cells with outgoing chemical edges: {len(fractions)}")
print(f"  all-inhibitory: {len(pure_inh)}   majority-inhibitory: {len(mostly_inh)}")
print(f"  mixed (<50% inh): {len(mixed)}   all-excitatory: {len(pure_exc)}")
print("\ntop 20 by inhibitory fraction:")
for frac, name, n in fractions[:20]:
    print(f"  {name:6} {frac:5.2f}  ({n} edges)")

gaba = [name for frac, name, _ in fractions if frac >= 0.5]
print(f"\nimplied GABAergic set ({len(gaba)}): {', '.join(sorted(gaba))}")
(ROOT / "docs" / "assets" / "sign_census.json").write_text(json.dumps(
    {"cells_with_edges": len(fractions),
     "pure_inhibitory": [n for f, n, _ in pure_inh],
     "implied_gabaergic": sorted(gaba),
     "fractions": [{"cell": n, "frac": round(f, 3), "edges": k} for f, n, k in fractions]},
    indent=1))
