"""Does the worm file carry usable synaptic signs? (Polarity cannot emerge without them.)"""
import json
from collections import Counter
from pathlib import Path

raw = json.loads((Path(__file__).resolve().parents[1] / "data" / "worm_connectome.json").read_text())
chem = raw["chemical"]
signs = Counter(round(float(r[3]), 3) for r in chem)
counts = Counter(round(float(r[3]), 3) for r in chem if float(r[2]) >= 3)
print("chemical edges:", len(chem), "sign values:", dict(signs))
print("edges with >=3 contacts:", sum(1 for r in chem if float(r[2]) >= 3), "signs:", dict(counts))
print("gap junctions:", len(raw["gap"]))
print("top keys:", list(raw.keys()))
print("neurons with role:", Counter(c.get("role", "?") for c in raw["neurons"]))
