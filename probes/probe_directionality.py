"""P0 from the 2026-11-09 review: is directionality already encoded in the wiring?

The insula mapping (Nature, 2026-10-21) says a region's role shows up as *direction*: the anterior
insula sends more than it receives, and that is what lets it modulate the posterior insula. ep4nn
has only ever used total weight and sign, never direction. If the c302 graph already separates
controllers from receivers, that is free information we have not used.

Measure per cell: out_weight / (out_weight + in_weight) over chemical contacts.
Read out by role, and test command interneurons against sensory cells.

    python probes/probe_directionality.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sheng/math/ep4nn")
raw = json.loads((ROOT / "data" / "worm_connectome.json").read_text())
names = [c["name"] for c in raw["neurons"]]
roles = [c.get("role", "unknown") for c in raw["neurons"]]

out_w = np.zeros(len(names))
in_w = np.zeros(len(names))
for pre, post, count, _sign in ((int(r[0]), int(r[1]), float(r[2]), float(r[3]))
                                for r in raw["chemical"]):
    out_w[pre] += count
    in_w[post] += count

total = out_w + in_w
ratio = np.where(total > 0, out_w / np.maximum(total, 1e-9), np.nan)

print(f"{'role':12} {'n':>3} {'median out/(out+in)':>20} {'mean':>7}")
rows = {}
for role in sorted(set(roles)):
    idx = np.array([i for i, r in enumerate(roles) if r == role])
    vals = ratio[idx]
    vals = vals[np.isfinite(vals)]
    rows[role] = {"n": int(len(idx)), "median": round(float(np.median(vals)), 3),
                  "mean": round(float(vals.mean()), 3)}
    print(f"{role:12} {len(idx):>3} {np.median(vals):>20.3f} {vals.mean():>7.3f}")

cmd = ratio[[i for i, r in enumerate(roles) if r == "command"]]
sen = ratio[[i for i, r in enumerate(roles) if r == "sensory"]]
cmd, sen = cmd[np.isfinite(cmd)], sen[np.isfinite(sen)]

# permutation test on the median difference: no scipy in this venv
rng = np.random.default_rng(0)
pool = np.concatenate([cmd, sen])
obs = float(np.median(cmd) - np.median(sen))
hits = 0
for _ in range(5000):
    perm = rng.permutation(pool)
    hits += abs(float(np.median(perm[:len(cmd)]) - np.median(perm[len(cmd):]))) >= abs(obs)
p = (hits + 1) / 5001

top = sorted(((float(ratio[i]), names[i], roles[i]) for i in range(len(names))
              if np.isfinite(ratio[i])), reverse=True)[:10]
print(f"\ncommand median {np.median(cmd):.3f} vs sensory {np.median(sen):.3f} "
      f"(diff {obs:+.3f}, permutation p={p:.4f})")
print("most control-like cells:", ", ".join(f"{n}({r[:3]}) {v:.2f}" for v, n, r in top))

out = ROOT / "docs" / "assets" / "directionality.json"
out.write_text(json.dumps({
    "probe": "probe_directionality.py",
    "by_role": rows,
    "command_vs_sensory": {"median_command": round(float(np.median(cmd)), 3),
                           "median_sensory": round(float(np.median(sen)), 3),
                           "diff": round(obs, 3), "permutation_p": round(p, 4)},
    "most_control_like": [{"cell": n, "role": r, "ratio": round(v, 3)} for v, n, r in top],
}, indent=1, ensure_ascii=False))
print(f"wrote {out}")
