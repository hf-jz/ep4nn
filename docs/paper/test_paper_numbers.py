"""Assert every number the paper prints against the artefact that produced it.

An audit that re-runs the pipeline still does not check transcription: a caption can carry an
invented ratio while every table entry stays correct. This asserts the printed strings against the
JSON dumps and the run logs, at the precision the paper prints, and the paper's build script runs it
before compiling, so a stale number cannot reach the PDF.

    python docs/paper/test_paper_numbers.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = (ROOT / "docs" / "paper" / "ep4nn.tex").read_text()
ASSETS = ROOT / "docs" / "assets"
RECEIPTS = ROOT / "docs" / "receipts"
FAILS: list[str] = []
check_count = 0


def printed(value: float, digits: int = 4) -> bool:
    """Is this value printed somewhere in the paper at this precision?"""
    forms = {f"{value:.{digits}g}", f"{value:.{digits}f}".rstrip("0").rstrip("."),
             f"{value:g}"}
    return any(f in TEX for f in forms if f)


def expect(condition: bool, message: str) -> None:
    global check_count
    check_count += 1
    if not condition:
        FAILS.append(message)


def main() -> None:
    scan = json.loads((ASSETS / "prereq_scan.json").read_text())
    rows = scan["rows"]
    best = max(rows, key=lambda r: r["spread"])
    expect(len(rows) == 45, f"the scan holds {len(rows)} operating points, the paper says forty-five")
    expect(printed(best["spread"], 3), f"best spread {best['spread']} not printed")
    expect(best["spread"] < 0.05, "the best spread is not below the growth gate")
    expect(min(r["collision_max"] for r in rows) >= 0.9979, "collision floor moved")

    audit = json.loads((ASSETS / "audit_fixes.json").read_text())
    g1 = audit["G1_certified_bound"]
    expect(printed(g1["default_efficacy"]["alpha_certified"], 6), "certified alpha (default) not printed")
    expect(printed(g1["declared_point_row_sum_0.5"]["alpha_certified"], 6), "certified alpha (declared) not printed")
    expect(printed(audit["G2_validity_domain"]["worst_alpha"], 3), "worst domain alpha not printed")
    expect(audit["G2_validity_domain"]["all_below_one"], "the certificate fails somewhere in the domain")
    seeds = [r["rho_K_numeric"] for r in audit["G5_random_seeds"]]
    expect(printed(min(seeds), 4) and printed(max(seeds), 4), "the five-seed range is not printed")
    expect(printed(sum(seeds) / len(seeds), 4), "the five-seed mean is not printed")

    closure = json.loads((ASSETS / "closure.json").read_text())
    pin = closure["closed_1_pinning"]
    worst = min(r["collision"] for r in pin["rows"])
    expect(worst >= 0.99, "a hand-built intervention now separates the pairs")
    expect(printed(pin["baseline_collision"], 6), "baseline collision not printed")
    drop = pin["baseline_collision"] - worst
    expect(drop < 0.02, "an intervention reaches the verdict margin; the paper says none does")
    expect(printed(round(drop, 4), 2), f"the ceiling drop {drop:.4f} is not printed")

    regime = json.loads((ASSETS / "regime.json").read_text())
    expect(len(regime["screen"]) == 24, f"the screen has {len(regime['screen'])} regimes, the paper says 24")
    expect(all(r["kept"] == 0 for r in regime["growth"]), "a growth regime kept a growth")

    r0 = next(r for r in regime["screen"] if r["row_sum"] == 0.5 and r["drive"] == 1.0 and r["hetero"] == 0.0)
    expect(printed(r0["mean_slope"], 2), "the regime table's slope is not printed")
    expect(printed(r0["mean_v"], 3), "the regime table's potential is not printed")

    names = ("grow-demo-default.json", "grow-demo-sensory.json", "grow-demo-interneuron.json")
    tone = [json.loads((RECEIPTS / n).read_text()) for n in names]
    expect(all(t["cost"]["ticks"] == 1200 for t in tone), "a readout receipt is not a 1200-tick run")
    expect(all(t["premise"].get("difference_source") == "state" for t in tone),
           "a receipt predates the state-targeting generator")
    readout_attempts = sum(t["cost"]["growths_kept"] + t["cost"]["growths_failed"] for t in tone)
    expect(readout_attempts == 90, f"the readout demos hold {readout_attempts} attempts, the paper says ninety")

    logs = sorted(RECEIPTS.glob("tone-scale*.log"))
    expect(len(logs) == 6, f"{len(logs)} tone arms on disk, the paper lists six")
    tone_attempts = 0
    for log in logs:
        line = next((ln for ln in log.read_text().splitlines() if ln.startswith("ticks=")), "")
        kept = int(line.split("kept=")[1].split()[0]) if "kept=" in line else 0
        failed = int(line.split("failed=")[1].split()[0]) if "failed=" in line else 0
        tone_attempts += kept + failed
        expect(kept == 0, f"{log.name} kept a growth")
    expect(tone_attempts == 90, f"the tone sweep holds {tone_attempts} attempts, the paper says ninety")

    hetero = json.loads((ASSETS / "hetero_calibrated.json").read_text())
    hetero_attempts = sum(r["kept"] + r["rolled_back"] for r in hetero["rows"])
    expect(hetero_attempts == 60, f"the heterogeneity arm holds {hetero_attempts} attempts, the paper says sixty")
    expect(all(r["kept"] == 0 for r in hetero["rows"]), "the heterogeneity arm kept a growth")

    regime_attempts = sum(r["kept"] + r["rolled_back"] for r in regime["growth"])
    expect(regime_attempts == 120, f"the regime arm holds {regime_attempts} attempts, the paper says one hundred and twenty")

    total = readout_attempts + tone_attempts + regime_attempts + hetero_attempts
    expect(total == 360, f"the attempts total {total}, the paper says three hundred and sixty")
    expect("three hundred and sixty" in TEX, "the abstract does not print the total")

    cap = json.loads((ASSETS / "capacity.json").read_text())
    written = cap["as the generator is"]
    held = cap["operating point held"]
    expect([r["neurons"] for r in held] == [302, 606, 1214], "the capacity arm's sizes moved")
    expect(written[1]["sweeps"] == 1024 and written[2]["sweeps"] == 1024,
           "the as-written arm settles again; the paper says it exhausts the budget")
    expect(held[1]["sweeps"] < 200 and held[2]["sweeps"] < 200,
           "the contract-held arm no longer settles")
    expect(held[1]["readout_spread"] > 0 and held[2]["readout_spread"] > 0,
           "the contract-held arm's answer is dead")
    expect(held[1]["settle_ms"] < held[0]["settle_ms"],
           "doubling is no longer cheaper than the baseline; the paper says it is")
    expect(printed(held[1]["settle_ms"], 3) and printed(held[2]["settle_ms"], 3),
           "the capacity settle times are not printed")
    expect(printed(held[1]["firing_fraction"], 3) and printed(written[2]["firing_fraction"], 3),
           "the firing shares are not printed")

    # the page count and the build are checked by build.sh; here only the content
    if FAILS:
        print(f"{len(FAILS)} of {check_count} checks failed")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print(f"paper numbers ok: {check_count} checks, {total} growth attempts accounted for")


if __name__ == "__main__":
    main()
