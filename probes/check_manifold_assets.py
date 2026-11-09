"""Cross-check the numbers typed into docs/实测台账.md §11 against the JSON assets."""
import json
from pathlib import Path

ROOT = Path("/Users/sheng/math/ep4nn")
rows = []
for path in sorted((ROOT / "docs" / "assets").glob("manifold*.json")):
    d = json.loads(path.read_text())
    rows.append((path.name, d.get("row_sum"), d.get("slope"), d.get("leak"),
                 d.get("bias_jitter"), d.get("gain_jitter"),
                 d["full"]["participation_ratio"], d["full"]["k90"],
                 d["readout"]["participation_ratio"], d.get("leakage"),
                 round(d.get("readout_share_of_var") or 0, 3)))
print(f"{'file':26} {'row':>5} {'slope':>6} {'leak':>5} {'jitter':>7} "
      f"{'PR':>6} {'k90':>4} {'roPR':>6} {'leak%':>6} {'roVar':>6}")
for r in rows:
    print(f"{r[0]:26} {str(r[1]):>5} {str(r[2]):>6} {str(r[3]):>5} "
          f"{str(r[4] if r[4] else ''):>7} {r[6]:>6} {r[7]:>4} {r[8]:>6} {str(r[9]):>6} {r[10]:>6}")
print(f"\n{len(rows)} assets")
