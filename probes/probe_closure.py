"""Closure probe: the three items the audit left open, each attacked with an experiment.

#1 growth benefit. Ninety attempts were rolled back. Is that a weakness of the generator
   or a property of the substrate? Test with the most favourable interventions available:
   direct edges from the context-carrying cells into the readout, at swept weight, and a
   hand-built repair neuron that reads the context and writes to the readout. If nothing in
   that family moves the collision past the verdict margin, the negative is structural, and
   the generator cannot be blamed for it.

#2 certificate scope. The bound was certified for the *computed* matrix; here the matrix
   itself is inflated outward by a stated relative slack, so the bound covers the exact
   model within that slack, and the residual gap is named (the model, not the arithmetic).

#3 no baseline. The 13-gene generator was never compared with anything. Two arms with the
   same budget: partners drawn from the difference-carrying neighbourhood (the generator) and
   partners drawn uniformly (a shuffle), plus a null arm that measures the collision's own
   drift with no growth at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a file, import the package

from cadence import Connectome
from cadence.brain import Brain
from cadence.learning import learning_neuron_model
from ep4nn.grow import EP4NN, GrowthConfig, _key, load_worm, operating_point

OUT: dict[str, object] = {}
LEVELS = np.linspace(0.2, 1.0, 5)
SLACK = 16 * np.finfo(float).eps          # relative slack for the outward inflation


# --- #2 -------------------------------------------------------------------

def certified_witness(K: np.ndarray, iters: int = 2000) -> dict:
    """rho(K_exact) <= alpha, with the matrix inflated outward by SLACK before the ratio."""
    Ku = K * (1.0 + SLACK)                # covers a few ulps of W, slope and dt arithmetic
    best = None
    for start in (np.ones(K.shape[0]), Ku.sum(axis=0) + 1e-12):
        h = np.asarray(start, float)
        for _ in range(iters):
            h = Ku @ h
            peak = float(np.max(h))
            if peak <= 0 or not np.isfinite(peak):
                break
            h = h / peak
        h = np.maximum(h, np.finfo(float).tiny)
        ratios = np.nextafter((Ku @ h) / h, np.inf)
        alpha = float(np.max(ratios))
        if best is None or alpha < best[0]:
            best = (alpha, h)
    alpha, h = best
    return {"alpha_certified": round(alpha, 6), "witness_holds": bool(np.all((Ku @ h) <= np.nextafter(alpha * h, np.inf))),
            "slack_relative": SLACK,
            "rho_numeric": round(float(np.max(np.abs(np.linalg.eigvals(K)))), 6)}


def tight_certificate() -> dict:
    conn = load_worm()
    drive = np.zeros((1, conn.n)); drive[0, :8] = 1.0
    out = {}
    for label, eff in (("default efficacy", None), ("declared point", operating_point(conn, 0.5))):
        kw = {} if eff is None else {"efficacy": eff}
        brain = Brain(conn, learning_neuron_model(), backend="cpu", **kw)
        nm = brain.neuron_model
        eq = brain.equilibrate(drive, budget=1024, chunk=32, tolerance=1e-4)
        v = np.asarray(eq.state.v).ravel()
        d = nm.slope_at(v)
        W = brain.dense()
        K = (1 - nm.dt) * np.eye(conn.n) + nm.dt * np.abs(W) * d[None, :]
        out[label] = certified_witness(K)
    return out


# --- #1 and #3 -------------------------------------------------------------

def collisions(ep: EP4NN, confused: list) -> float:
    return max(ep._collision(code, p, ctx) for code, p, ctx in confused)


def growth_arms(attempts: int = 30) -> dict:
    """Same budget, three arms: generator partners, shuffled partners, and no growth."""
    conn = load_worm()
    # seed the store the collision rule reads: same present, contexts 0..4
    codes = []
    for lv in LEVELS:
        code, _ = EP4NN(conn, GrowthConfig())._settle(
            EP4NN(conn, GrowthConfig())._drive(("odour_A", 1.0), ("pain", float(lv))))
        codes.append(code)
    arms = {}
    drives = [EP4NN(conn, GrowthConfig())._drive(("odour_A", 1.0), ("pain", float(lv)))
              for lv in LEVELS]
    setup = EP4NN(conn, GrowthConfig())
    for i, c in enumerate(codes):
        setup._remember(c, 0, i, drives[i])
    confused = [(codes[0], 0, len(LEVELS) - 1), (codes[-1], 0, 0)]
    for label, locality in (("generator partners", 0.7), ("shuffled partners", 0.0)):
        ep = EP4NN(load_worm(), GrowthConfig(locality=locality))
        for i, c in enumerate(codes):
            ep._remember(c, 0, i, drives[i])
        before = collisions(ep, confused)
        deltas = []
        for i in range(attempts):
            ep.ticks = i                      # a fresh draw per attempt
            entry = ep.try_grow(confused)
            if entry["event"] == "budget":
                continue
            deltas.append(entry["after"] - entry["before"])
        arms[label] = {"before": round(before, 6), "attempts": len(deltas),
                       "mean_delta": round(float(np.mean(deltas)), 6),
                       "best_delta": round(float(np.min(deltas)), 6),
                       "kept": int(ep.growths_kept), "margin": ep.config.improve_margin}
    null = EP4NN(load_worm(), GrowthConfig())
    for i, c in enumerate(codes):             # the same store, measured with the same instrument
        null._remember(c, 0, i, drives[i])
    before = null._fresh_collision()
    null_deltas = [null._fresh_collision() - before for _ in range(attempts)]
    arms["no growth (drift floor)"] = {"before": round(before, 6), "attempts": len(null_deltas),
                                      "mean_delta": round(float(np.mean(null_deltas)), 6),
                                      "best_delta": round(float(np.min(null_deltas)), 6)}
    return arms


def wide_efficacy(conn: Connectome, wide: Connectome, w: float, touch) -> tuple[np.ndarray, int]:
    """Efficacy of the widened graph, by synapse key: existing keys keep the declared value,
    touched keys take ``w``, and anything else is new and starts silent."""
    base = {int(k): float(e) for k, e in
            zip(_key(np.asarray(conn.pre), np.asarray(conn.post), conn.n),
                operating_point(conn, 0.5))}
    eff = np.zeros(wide.synapses)
    touched = 0
    for i, (a, b) in enumerate(zip(np.asarray(wide.pre), np.asarray(wide.post))):
        k = int(b) * wide.n + int(a)
        if touch(int(a), int(b)):
            eff[i] = w
            touched += 1
        elif k in base:
            eff[i] = base[k]
    return eff, touched


def pinning_test() -> dict:
    """The most favourable interventions the substrate allows: direct context->readout edges,
    and a hand-built repair neuron that reads the context cells and writes to the readout."""
    conn = load_worm()
    pops = conn.populations
    pain = set(pops["pain"])
    readout = sorted(set(pops["forward"]) | set(pops["reverse"]))
    rows = []
    for w in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        pre = list(conn.pre) + [p for p in pain for _ in readout]
        post = list(conn.post) + [r for _ in pain for r in readout]
        sign = list(conn.sign) + [1.0] * (len(pain) * len(readout))
        count = list(conn.count) + [1.0] * (len(pain) * len(readout))
        wide = Connectome.from_synapses(conn.n, pre=pre, post=post, count=count, sign=sign,
                                       populations=pops, label="worm+direct")
        eff, touched = wide_efficacy(conn, wide, w, lambda a, b: a in pain and b in readout)
        rows.append({"intervention": f"direct pain->readout edges, weight {w:g}",
                     "collision": _pair_collision(wide, eff, readout), "edges_touched": touched})
    for w in (0.5, 2.0, 10.0):                # one hand-built repair neuron
        fresh = conn.n
        pre = list(conn.pre) + list(pain) + [fresh] * len(readout)
        post = list(conn.post) + [fresh] * len(pain) + readout
        sign = list(conn.sign) + [1.0] * (len(pain) + len(readout))
        count = list(conn.count) + [1.0] * (len(pain) + len(readout))
        wide = Connectome.from_synapses(conn.n + 1, pre=pre, post=post, count=count, sign=sign,
                                       populations=pops, label="worm+repair")
        eff, touched = wide_efficacy(conn, wide, w,
                                     lambda a, b: a in pain and b == fresh
                                     or a == fresh and b in readout)
        rows.append({"intervention": f"hand-built repair neuron, weight {w:g}",
                     "collision": _pair_collision(wide, eff, readout), "edges_touched": touched})
    return {"baseline_collision": _pair_collision(conn, operating_point(conn, 0.5), readout),
            "rows": rows}


def _pair_collision(conn: Connectome, eff: np.ndarray, readout: list[int]) -> float:
    brain = Brain(conn, learning_neuron_model(), backend="cpu", efficacy=eff)
    codes = []
    for lv in LEVELS:
        drive = np.zeros((1, conn.n))
        for name, level in (("odour_A", 1.0), ("pain", float(lv))):
            drive[0, list(conn.populations[name])] = level
        eq = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4)
        codes.append(np.asarray(eq.state.activation).ravel()[readout])
    codes = np.stack(codes)
    unit = codes / np.maximum(np.linalg.norm(codes, axis=1)[:, None], 1e-12)
    cos = unit @ unit.T
    return round(max(abs(float(cos[0, j])) for j in range(1, len(LEVELS))), 6)


def main() -> None:
    OUT["closed_2_certificate"] = tight_certificate()
    OUT["closed_1_and_3_growth_arms"] = growth_arms()
    OUT["closed_1_pinning"] = pinning_test()
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "closure.json"
    dump.write_text(json.dumps(OUT, indent=1, default=str) + "\n")
    c = OUT["closed_2_certificate"]
    print(f"#2 certified alpha (outward-inflated by {SLACK:.2e}): "
          f"default {c['default efficacy']['alpha_certified']}, declared {c['declared point']['alpha_certified']}; "
          f"witness holds {c['default efficacy']['witness_holds']}/{c['declared point']['witness_holds']}")
    print("#1/#3 arms:")
    for k, v in OUT["closed_1_and_3_growth_arms"].items():
        print(f"   {k:<24} before {v['before']:.6f} mean delta {v['mean_delta']:+.6f} "
              f"best delta {v['best_delta']:+.6f}")
    p = OUT["closed_1_pinning"]
    print(f"#1 pinning: baseline collision {p['baseline_collision']}")
    for r in p["rows"]:
        print(f"   {r['intervention']:<45} collision {r['collision']:.6f} ({r['edges_touched']} edges)")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
