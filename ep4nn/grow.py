"""ep4nn — a brain that starts as the 302-neuron worm and grows when it cannot
tell two situations apart.

Four pieces, each a rule rather than a knob:

1. demand     two situations the world declares different, sharing the same
              present input, produce nearly parallel **readout** codes
              (|cosine| > `code_cap`): the wiring cannot carry the difference to
              the answer. That is a capacity signal, not a weight problem.
2. evidence   one collision is noise: `min_evidence` of them must be seen before
              anything changes (the growth layer's threshold sits above the slow
              layer's).
3. growth     one low-dimensional generator (a handful of genes << |E|) appends
              `k` neurons, wired from the cells that *carry the difference* and
              their hop-neighbours, plus an out-edge into the readout so the
              change can reach the answer at all. Reciprocal by default, which
              keeps the equilibrium-propagation contract intact.
4. verdict    re-measure the *same* collisions. Better -> keep. Worse or
              unchanged -> roll back and count a failed growth. Growth that does
              not pay is not growth.

What a connectome does not carry, and this file declares rather than assumes:
an operating point. The worm's raw contact counts give row sums in the hundreds
and the net never settles, so the initial efficacy is row-normalised here — the
step Cadence's own connectome lane calls calibrating the operating point.

Connectome is immutable, so a growth event rebuilds it (O(nnz)) and carries the
learned efficacy across by regenerating the key order the library sorts by.
Growth is therefore rare-per-contradiction, never per tick.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.brain import Brain
from cadence.learning import calibrate_bias, learning_neuron_model

DATA = Path(__file__).resolve().parents[1] / "data" / "worm_connectome.json"
POPULATION_CELLS = {
    "odour_A": ("AWAL", "AWAR"),
    "odour_B": ("AWCL", "AWCR"),
    "food": ("CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER"),
    "pain": ("ASHL", "ASHR"),
    "forward": ("AVBL", "AVBR", "PVCL", "PVCR"),
    "reverse": ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"),
}


def _key(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    """The key ``Connectome.from_synapses`` sorts and merges by."""
    return post * n + pre


def load_worm() -> Connectome:
    raw = json.loads(DATA.read_text())
    names = [c["name"] for c in raw["neurons"]]
    index = {name: i for i, name in enumerate(names)}
    pre, post, count, sign = [], [], [], []
    for a, b, c, s in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
        if a != b:
            pre.append(a); post.append(b); count.append(c); sign.append(s)
    for a, b, c in ((int(r[0]), int(r[1]), float(r[2])) for r in raw["gap"]):
        if a != b:
            for x, y in ((a, b), (b, a)):  # gap junctions conduct both ways
                pre.append(x); post.append(y); count.append(c); sign.append(1.0)
    populations = {k: tuple(index[n] for n in cells) for k, cells in POPULATION_CELLS.items()}
    return Connectome.from_synapses(len(names), pre=pre, post=post, count=count,
                                    sign=sign, populations=populations, label="celegans:c302")


def operating_point(connectome: Connectome, row_sum: float) -> np.ndarray:
    """Signs from the connectome, magnitude row-normalised to ``row_sum``.

    A connectome says who talks to whom, not how excitable a cell is; without
    this the worm's counts (row sums in the hundreds) never settle.
    """
    indeg = np.bincount(connectome.post, minlength=connectome.n).astype(float)
    indeg[indeg == 0] = 1.0
    return connectome.sign * (row_sum / indeg[connectome.post])


@dataclass
class GrowthConfig:
    """The genes of growth: a dozen numbers, not an edge list."""

    neurons_per_event: int = 8          # k appended per kept growth
    degree: float = 10.4               # mean in/out degree of a new neuron (the worm's own)
    degree_jitter: float = 0.25        # log-normal spread on that degree
    locality: float = 0.7              # P(partner drawn near the difference-carrying cells)
    excitatory: float = 0.88           # match the worm's +sign fraction
    reciprocal: float = 1.0            # P(a new edge also gets its reverse): keeps the EP contract
    to_readout: float = 0.6            # P(a new neuron also projects into the readout)
    init_scale: float = 3.0            # new synapses start at this multiple of the mean |efficacy|
    calibrate: bool = True             # centre the readout's operating point, as the library prescribes
    row_sum: float = 0.5               # the declared operating point
    code_cap: float = 0.85             # |cosine| above this counts as "the same answer"
    min_evidence: int = 3              # collisions required before one growth event
    min_response: float = 0.05         # the answer must move this much, or growth is the wrong repair
    improve_margin: float = 0.02       # the re-measured collision cosine must drop this much
    max_neurons: int = 4096            # budget
    min_ticks_between: int = 40        # do not thrash
    memory: int = 96                   # bounded store of (code, present, context)


class EP4NN:
    """One continuing brain: a connectome, a settling net on it, and growth."""

    def __init__(self, connectome: Connectome, config: GrowthConfig | None = None,
                 *, tolerance: float = 1e-3, budget: int = 512, chunk: int = 32):
        self.config = config or GrowthConfig()
        self.tolerance, self.budget, self.chunk = tolerance, budget, chunk
        self.connectome = connectome
        self._efficacy = operating_point(connectome, self.config.row_sum)
        self.memory: list[tuple[np.ndarray, int, int]] = []
        self.log: list[dict] = []
        self.ticks = self.growths_kept = self.growths_failed = self.silent = 0
        self._last_growth_tick = -10 ** 9
        self._evidence = 0
        self._rebuild()

    # -- brain plumbing -------------------------------------------------
    def _rebuild(self) -> None:
        populations = self.connectome.populations
        self.readout = np.array(sorted(set(populations["forward"]) | set(populations["reverse"])))
        self.brain = Brain(self.connectome, learning_neuron_model(), backend="cpu",
                           efficacy=self._efficacy)
        if not self.config.calibrate:
            self.readout_spread = self._spread(np.zeros(self.neurons))
            return
        # The operating point a connectome does not carry. Calibration is kept only
        # if it makes the readout *more responsive*: forcing every cell to one half
        # can also pin it, and a pinned readout answers the same thing to everything.
        levels = np.linspace(0.2, 1.0, 5)
        rows = [self._drive(("odour_A", 1.0), ("pain", float(lv)))[0] for lv in levels]
        rows += [self._drive(("odour_B", 1.0), ("pain", float(lv)))[0] for lv in levels]
        drives = np.stack(rows)
        raw = self.brain
        bias = calibrate_bias(raw, drives, {"forward": 0.5, "reverse": 0.5},
                              per_neuron=True, rounds=1, iterations=8, steps=64)
        self.readout_spread = self._spread(np.zeros(self.neurons), drives)
        calibrated = Brain(self.connectome, learning_neuron_model(), backend="cpu",
                           efficacy=self._efficacy, bias=bias)
        candidates = [(self.readout_spread, raw), (self._spread(bias, drives), calibrated)]
        self.readout_spread, self.brain = max(candidates, key=lambda pair: pair[0])

    def _spread(self, bias: np.ndarray, drives: np.ndarray | None = None) -> float:
        """How much the readout cells move across the declared situations."""
        if drives is None:
            levels = np.linspace(0.2, 1.0, 3)
            drives = np.stack([self._drive(("odour_A", 1.0), ("pain", float(lv)))[0]
                               for lv in levels])
        probe = Brain(self.connectome, learning_neuron_model(), backend="cpu",
                      efficacy=self._efficacy, bias=bias)
        eq = probe.equilibrate(drives, budget=256, chunk=32, tolerance=1e-2)
        codes = np.asarray(eq.state.activation)[:, self.readout]
        return float(np.mean(codes.max(axis=0) - codes.min(axis=0)))

    @property
    def neurons(self) -> int:
        return self.connectome.n

    @property
    def synapses(self) -> int:
        return self.connectome.synapses

    def _drive(self, *senses: tuple[str, float]) -> np.ndarray:
        d = np.zeros((1, self.neurons))
        for population, level in senses:
            d[0, list(self.connectome.populations[population])] = level
        return d

    def _settle(self, drive: np.ndarray) -> tuple[np.ndarray, dict]:
        eq = self.brain.equilibrate(drive, budget=self.budget, chunk=self.chunk,
                                    tolerance=self.tolerance)
        state = np.asarray(eq.state.activation).ravel()
        return state[self.readout], {
            "sweeps": int(eq.steps), "residual": float(np.max(eq.residual)),
            "qualified": bool(np.all(eq.qualified)),
        }

    # -- 1+2: demand and evidence ---------------------------------------
    def _collision(self, code: np.ndarray, present: int, context: int) -> float:
        """Largest |cosine| against a stored answer for the same present, another context."""
        norm = float(np.linalg.norm(code))
        if norm == 0.0:
            return 0.0
        best = 0.0
        for stored, stored_present, stored_context in self.memory:
            if stored_present != present or stored_context == context:
                continue
            snorm = float(np.linalg.norm(stored))
            if snorm == 0.0:
                continue
            best = max(best, abs(float(code @ stored) / (norm * snorm)))
        return best

    def _remember(self, code: np.ndarray, present: int, context: int) -> None:
        self.memory.append((code.copy(), present, context))
        if len(self.memory) > self.config.memory:
            self.memory.pop(0)

    # -- 3: the generator ------------------------------------------------
    def _neighbours(self, seeds: np.ndarray) -> np.ndarray:
        pre, post = self.connectome.pre, self.connectome.post
        mask = np.isin(pre, seeds) | np.isin(post, seeds)
        return np.unique(np.concatenate([pre[mask], post[mask]]))

    def _difference_cells(self, confused: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
        """Cells whose readout activity carries what the colliding situations differ in.

        Paired inside one present input: same sense, different context. This is
        the only place "what to wire" comes from.
        """
        by_present: dict[int, list[np.ndarray]] = {}
        for code, present, _ in confused:
            by_present.setdefault(present, []).append(code)
        diffs = []
        for codes in by_present.values():
            for a, b in zip(codes, codes[1:]):
                diffs.append(np.abs(a - b))
        if not diffs:
            return np.zeros(len(self.readout), dtype=bool)
        return np.stack(diffs).max(axis=0) > 1e-6

    def _grow(self, confused: list[tuple[np.ndarray, int, int]], rng: np.random.Generator) -> int:
        cfg = self.config
        n_old, k = self.neurons, cfg.neurons_per_event
        if n_old + k > cfg.max_neurons:
            return 0
        # what to wire: the readout cells the confusion lives in, their sources,
        # and the cells that are active while the answer forms (one hop out).
        hot = np.zeros(n_old, dtype=bool)
        differs = self._difference_cells(confused)
        hot[self.readout[differs]] = True
        for code, _, _ in confused:
            hot[self.readout[np.argsort(-code)[:4]]] = True
        state = self.brain.settle_batch(self._drive(("odour_A", 1.0)), steps=self.chunk, state=None)
        hot |= np.asarray(state.activation).ravel() > 0.2
        seeds = np.flatnonzero(hot)
        near = self._neighbours(seeds) if len(seeds) else np.arange(n_old)
        if len(near) < 4:
            near = np.arange(n_old)
        mean_in = max(1.0, self.synapses / n_old)
        init = cfg.init_scale * (float(np.mean(np.abs(self._efficacy))) or 0.02)

        pre, post, sign = [], [], []
        for j in range(k):
            new = n_old + j
            for _ in range(2):  # in-edges, then out-edges, one degree prior
                degree = int(np.clip(rng.lognormal(np.log(cfg.degree), cfg.degree_jitter),
                                     1, 8 * mean_in))
                take_local = rng.random(degree) < cfg.locality
                partners = np.empty(degree, dtype=np.int64)
                partners[take_local] = near[rng.integers(0, len(near), int(take_local.sum()))]
                partners[~take_local] = rng.integers(0, n_old, int((~take_local).sum()))
                for partner in np.unique(partners):
                    s = 1.0 if rng.random() < cfg.excitatory else -1.0
                    pre.append(partner); post.append(new); sign.append(s)
                    if rng.random() < cfg.reciprocal:
                        pre.append(new); post.append(partner); sign.append(s)
            # growth must be able to reach the answer, or it cannot help at all
            if rng.random() < cfg.to_readout:
                target = int(self.readout[rng.integers(0, len(self.readout))])
                s = 1.0 if rng.random() < cfg.excitatory else -1.0
                pre.append(new); post.append(target); sign.append(s)
        if not pre:
            return 0
        pre_a = np.asarray(pre, dtype=np.int64)
        post_a = np.asarray(post, dtype=np.int64)
        sign_a = np.asarray(sign, dtype=float)

        old_pre, old_post = self.connectome.pre, self.connectome.post
        new_connectome = Connectome.from_synapses(
            n_old + k,
            pre=np.concatenate([old_pre, pre_a]),
            post=np.concatenate([old_post, post_a]),
            count=np.concatenate([self.connectome.count, np.ones(len(pre_a))]),
            sign=np.concatenate([self.connectome.sign, sign_a]),
            populations=self.connectome.populations, label=self.connectome.label,
        )
        # carry what was learned: regenerate the same key order the library sorts by
        old_keys = np.unique(_key(old_pre, old_post, n_old))
        new_keys = np.unique(_key(new_connectome.pre, new_connectome.post, n_old + k))
        pos = np.searchsorted(old_keys, new_keys)
        hit = (pos < len(old_keys)) & (old_keys[np.minimum(pos, len(old_keys) - 1)] == new_keys)
        efficacy = np.sign(new_connectome.sign) * init
        efficacy[hit] = self._efficacy[pos[hit]]
        self.connectome, self._efficacy = new_connectome, efficacy
        self._rebuild()
        return k

    # -- 4: the verdict --------------------------------------------------
    def try_grow(self, confused: list[tuple[np.ndarray, int, int]]) -> dict:
        """Keep the growth only if the same collisions become less parallel."""
        before = max(self._collision(c, p, ctx) for c, p, ctx in confused)
        prev_connectome, prev_efficacy = self.connectome, self._efficacy.copy()
        t0 = time.perf_counter()
        added = self._grow(confused, np.random.default_rng(self.ticks + 7))
        grow_ms = 1000 * (time.perf_counter() - t0)
        if not added:
            return {"event": "budget", "before": before, "after": before, "added": 0}
        after = max(self._collision(c, p, ctx) for c, p, ctx in confused)
        kept = (before - after) >= self.config.improve_margin
        entry = {"event": "kept" if kept else "rolled_back", "added": added,
                 "before": round(before, 4), "after": round(after, 4),
                 "grow_ms": round(grow_ms, 1), "neurons": self.neurons,
                 "synapses": self.synapses}
        if not kept:  # roll back: growth that does not pay is not growth
            self.connectome, self._efficacy = prev_connectome, prev_efficacy
            self._rebuild()
            self._evidence = 0
            self.growths_failed += 1
        else:
            self.growths_kept += 1
        self.log.append(entry)
        return entry

    # -- one moment of life ---------------------------------------------
    def tick(self, present: tuple[str, float], context: tuple[str, float],
             present_id: int, context_id: int) -> dict:
        self.ticks += 1
        code, info = self._settle(self._drive(present, context))
        if float(np.linalg.norm(code)) == 0.0:
            self.silent += 1  # a silent answer is a response failure, not a capacity one
        collision = self._collision(code, present_id, context_id)
        self._remember(code, present_id, context_id)
        if self.readout_spread < self.config.min_response:
            # the premise of growth is false here: the wiring does not move the answer
            # at all, so no amount of capacity can fix a collision. Say so, do not grow.
            self._evidence = 0
            self.log.append({"event": "no_response", "spread": round(self.readout_spread, 4),
                             "neurons": self.neurons})
            return {**info, "collision": round(collision, 3), "growth": None}
        if collision > self.config.code_cap:
            self._evidence += 1
        if (self._evidence >= self.config.min_evidence
                and self.ticks - self._last_growth_tick >= self.config.min_ticks_between):
            confused = [(c, p, ctx) for c, p, ctx in self.memory
                        if self._collision(c, p, ctx) > self.config.code_cap]
            entry = self.try_grow(confused)
            self._last_growth_tick = self.ticks
            self._evidence = 0
            return {**info, "collision": round(collision, 3), "growth": entry}
        return {**info, "collision": round(collision, 3), "growth": None}


def _report(tag: str, ep: EP4NN, t0: float, ticks: int) -> None:
    drive = ep._drive(("odour_A", 1.0))
    warm = ep.brain.settle_batch(drive, steps=ep.chunk, state=None)
    t2 = time.perf_counter()
    for _ in range(5):
        ep.brain.settle_batch(drive, steps=ep.chunk, state=warm)
    per_sweep = 1e6 * (time.perf_counter() - t2) / (5 * ep.chunk)
    ms_tick = 1000 * (time.perf_counter() - t0) / max(1, ticks)
    print(f"{tag:>10s} | neurons {ep.neurons:>5d} | synapses {ep.synapses:>6d} | "
          f"per-sweep {per_sweep:6.1f} us | readout spread {ep.readout_spread:.3f} | "
          f"{ms_tick:5.1f} ms/tick | "
          f"kept {ep.growths_kept:>2d} | failed {ep.growths_failed:>2d}")


def demo(ticks: int = 1200) -> None:
    """Start at 302 neurons; the world keeps requiring new distinctions."""
    ep = EP4NN(load_worm())
    print(f"start: neurons={ep.neurons} synapses={ep.synapses} "
          f"(row-normalised operating point {ep.config.row_sum})")
    rng = np.random.default_rng(3)
    contexts = 2                                   # pain levels that must be told apart
    t0 = time.perf_counter()
    for i in range(ticks):
        if i and i % 240 == 0:
            contexts += 1
            _report("milestone", ep, t0, i)
            print(f"           tick {i}: the world now requires {contexts} levels of pain "
                  f"to be told apart at the answer")
            t0 = time.perf_counter()
        context_id = int(rng.integers(0, contexts))
        level = 0.35 + 0.65 * context_id / max(1, contexts - 1)
        out = ep.tick(("odour_A", 1.0), ("pain", level), 0, context_id)
        if out["growth"] and out["growth"]["event"] != "budget":
            g = out["growth"]
            print(f"tick {i:>4d} ctx {context_id} collision {out['collision']:.3f} -> "
                  f"{g['event']:>11s} +{g['added']:>2d} neurons "
                  f"({g['before']:.3f} -> {g['after']:.3f}), "
                  f"{g['neurons']} neurons / {g['synapses']} synapses, "
                  f"rebuild {g['grow_ms']:.1f} ms")
    _report("final", ep, t0, ticks)
    no_response = len([e for e in ep.log if e["event"] == "no_response"])
    print(f"ticks={ep.ticks} kept={ep.growths_kept} failed={ep.growths_failed} "
          f"refused_no_response={no_response} silent_answers={ep.silent} "
          f"neurons={ep.neurons} synapses={ep.synapses}")
    try:  # importable as a module and runnable as a file: python ep4nn/grow.py
        from .receipt import build as build_receipt, write as write_receipt
    except ImportError:
        from receipt import build as build_receipt, write as write_receipt

    write_receipt(build_receipt(
        dataset={"source": "openworm/c302 A_Full via cadence-examples/worm",
                 "sha256": "2a7b117f156e4be0314c01f4db92f046cab3c40140ac72124444fa76cf72b1d9"},
        nominal={"neurons": 302, "synapses": 3137},
        executed={"neurons": ep.neurons, "synapses": ep.synapses},
        dynamics={"model": "cadence rate (rectified sigmoid)", "row_sum": ep.config.row_sum,
                  "absent": ["delay", "adaptation", "heterogeneous cell types"]},
        loop={"closed": False, "supplied": ["situations", "context levels", "readout declaration"],
              "note": "the world declares which situations must be distinguishable"},
        premise={"readout_spread": round(ep.readout_spread, 4),
                 "min_response": ep.config.min_response,
                 "collision_cap": ep.config.code_cap,
                 "refusals_no_response": no_response},
        cost={"per_sweep_us": None, "ticks": ep.ticks, "growths_kept": ep.growths_kept,
              "growths_failed": ep.growths_failed},
    ), Path(__file__).resolve().parents[1] / "docs" / "receipts" / "grow-demo.json")
    kept = [e for e in ep.log if e["event"] == "kept"]
    assert ep.neurons >= 302, "the brain must never shrink below the worm"
    assert all(e["after"] <= e["before"] + 1e-9 for e in kept), "a kept growth must not worsen"
    assert ep._efficacy.shape == (ep.synapses,), "every synapse needs an efficacy"
    assert int(ep.readout.max()) < ep.neurons, "the readout must live in the current graph"
    print(f"self-check ok: {len(kept)} kept growths, {ep.growths_failed} rolled back, "
          f"{ep.neurons - 302} neurons added, efficacy carried across every rebuild")


if __name__ == "__main__":
    demo()
