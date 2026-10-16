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
import sys
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


def load_worm(tone: bool = False) -> Connectome:
    """The worm wiring, optionally with two extra input cells that drive nothing.

    ``tone`` is the unit test of structural plasticity: the world's context lives in two cells that
    reach no one, so the situations differ in the network and not in the answer. Supplying that path
    is exactly the repair growth is supposed to be able to make.
    """
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
    # the wiring's own cell classes (the only class metadata a connectome file carries)
    roles = [c.get("role", "unknown") for c in raw["neurons"]]
    for role in sorted(set(roles)):
        populations[f"class_{role}"] = tuple(i for i, r in enumerate(roles) if r == role)
    n = len(names)
    if tone:
        populations["context_tone"] = (n, n + 1)      # two cells with no outgoing edge at all
        n += 2
    return Connectome.from_synapses(n, pre=pre, post=post, count=count,
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
    min_state_motion: float = 1e-3     # the declared pair must differ somewhere in the network, else the two
                                       # situations are one situation and no wiring can separate them
    min_pair_motion: float = 0.0       # diagnostic: how far apart the answers of a pair sit, relative to size
    readout: tuple[str, ...] = ("forward", "reverse")   # populations that constitute the declared answer
    difference_source: str = "state"   # "state": cells whose activity differs anywhere; "readout": only the answer
    context: str = "pain"              # the population the world varies to make a pair
    drive_scale: float = 1.0           # scales every input level: the regime knob the protocol selects
    hetero_spread: float = 0.0         # per-neuron slope/threshold spread; 0 is the library's scalar model
    hetero_seed: int = 3
    difference_fraction: float = 0.06  # top fraction of differing cells the generator wires from
    difference_share: float = 0.6      # share of a new neuron's in-edges drawn from those cells
    readout_sign: float = 1.0          # sign of the new neuron's projection into the answer; -1 is inhibition
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
        self._drives: dict[tuple[int, int], np.ndarray] = {}   # (present, context) -> drive
        self._full: dict[tuple[int, int], np.ndarray] = {}     # (present, context) -> whole state
        self.log: list[dict] = []
        self.ticks = self.growths_kept = self.growths_failed = self.silent = 0
        self._last_growth_tick = -10 ** 9
        self._evidence = 0
        self._rebuild()

    # -- brain plumbing -------------------------------------------------
    def _rebuild(self) -> None:
        populations = self.connectome.populations
        self.readout = np.array(sorted({i for name in self.config.readout
                                        for i in populations[name]}))
        self.brain = Brain(self.connectome, self._model(), backend="cpu",
                           efficacy=self._efficacy)
        if not self.config.calibrate:
            self.readout_spread = self._spread(np.zeros(self.neurons))
            self.pair_motion = self._pair_motion()
            self.state_motion = self._state_motion()
            return
        # The operating point a connectome does not carry. Calibration is kept only
        # if it makes the readout *more responsive*: forcing every cell to one half
        # can also pin it, and a pinned readout answers the same thing to everything.
        levels = np.linspace(0.2, 1.0, 5)
        ctx = self.config.context
        rows = [self._drive(("odour_A", 1.0), (ctx, float(lv)))[0] for lv in levels]
        rows += [self._drive(("odour_B", 1.0), (ctx, float(lv)))[0] for lv in levels]
        drives = np.stack(rows)
        raw = self.brain
        bias = calibrate_bias(raw, drives, {"forward": 0.5, "reverse": 0.5},
                              per_neuron=True, rounds=1, iterations=8, steps=64)
        self.readout_spread = self._spread(np.zeros(self.neurons), drives)
        calibrated = Brain(self.connectome, self._model(), backend="cpu",
                           efficacy=self._efficacy, bias=bias)
        candidates = [(self.readout_spread, raw), (self._spread(bias, drives), calibrated)]
        self.readout_spread, self.brain = max(candidates, key=lambda pair: pair[0])
        self.pair_motion = self._pair_motion()
        self.state_motion = self._state_motion()

    def _state_motion(self) -> float:
        """How far apart a declared pair is *anywhere in the network*, relative to its own size.

        This is the honest prerequisite for a structural repair. Asking whether the readout already
        moves refuses precisely the case growth exists for: a distinction that is present in the
        network and has no path to the declared answer.
        """
        best = 0.0
        by_present: dict[int, list[np.ndarray]] = {}
        for (present, _context), state in self._full.items():
            by_present.setdefault(present, []).append(state)
        for states in by_present.values():
            for a, b in zip(states, states[1:]):
                scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-12)
                best = max(best, float(np.linalg.norm(a - b)) / scale)
        return best

    def _pair_motion(self) -> float:
        """Diagnostic: how far apart a declared pair's answers sit, relative to their own size.

        The pair is the unit the world declares: same present, different context. The
        relative form is the one that matters, because the gate has to tell "the two
        answers are the same object" from "they differ by a little and a repair could
        amplify that". An aggregate spread over all situations cannot: it can move while
        every pair stands still, and a mean over many readout cells dilutes the few cells
        that carry the difference.
        """
        codes = self._situation_codes()
        half = len(codes) // 2
        motions = []
        for start in (0, half):
            for i in range(start, start + half):
                for j in range(i + 1, start + half):
                    scale = max(float(np.linalg.norm(codes[i])), float(np.linalg.norm(codes[j])), 1e-12)
                    motions.append(float(np.linalg.norm(codes[i] - codes[j])) / scale)
        return float(np.mean(motions)) if motions else 0.0

    def _situation_codes(self) -> np.ndarray:
        """The declared readout's answer to every declared situation."""
        levels = np.linspace(0.2, 1.0, 5)
        ctx = self.config.context
        rows = [self._drive(("odour_A", 1.0), (ctx, float(lv)))[0] for lv in levels]
        rows += [self._drive(("odour_B", 1.0), (ctx, float(lv)))[0] for lv in levels]
        eq = self.brain.equilibrate(np.stack(rows), budget=self.budget, chunk=self.chunk,
                                    tolerance=self.tolerance)
        return np.asarray(eq.state.activation)[:, self.readout]

    def _spread(self, bias: np.ndarray, drives: np.ndarray | None = None) -> float:
        """How much the readout cells move across the declared situations."""
        if drives is None:
            levels = np.linspace(0.2, 1.0, 3)
            drives = np.stack([self._drive(("odour_A", 1.0), (self.config.context, float(lv)))[0]
                               for lv in levels])
        probe = Brain(self.connectome, self._model(), backend="cpu",
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

    def _model(self):
        """The neuron model for this brain: the library's, or a per-neuron heterogeneous one."""
        if self.config.hetero_spread <= 0.0:
            return learning_neuron_model()
        from .hetero import HeteroRateModel
        return HeteroRateModel(n=self.neurons, spread=self.config.hetero_spread,
                               seed=self.config.hetero_seed)

    def _drive(self, *senses: tuple[str, float]) -> np.ndarray:
        d = np.zeros((1, self.neurons))
        for population, level in senses:
            d[0, list(self.connectome.populations[population])] = level * self.config.drive_scale
        return d

    def _settle_full(self, drive: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        eq = self.brain.equilibrate(drive, budget=self.budget, chunk=self.chunk,
                                    tolerance=self.tolerance)
        state = np.asarray(eq.state.activation).ravel()
        return state[self.readout], state, {
            "sweeps": int(eq.steps), "residual": float(np.max(eq.residual)),
            "qualified": bool(np.all(eq.qualified)),
        }

    def _settle(self, drive: np.ndarray) -> tuple[np.ndarray, dict]:
        code, _state, info = self._settle_full(drive)
        return code, info

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

    def _remember(self, code: np.ndarray, present: int, context: int,
                  drive: np.ndarray | None = None, state: np.ndarray | None = None) -> None:
        self.memory.append((code.copy(), present, context))
        if len(self.memory) > self.config.memory:
            self.memory.pop(0)
        if state is not None:
            self._full[(present, context)] = np.asarray(state, float).copy()
            while len(self._full) > self.config.memory:
                self._full.pop(next(iter(self._full)))
        if drive is not None:
            self._drives[(present, context)] = np.asarray(drive, float).copy()
            while len(self._drives) > self.config.memory:
                self._drives.pop(next(iter(self._drives)))

    def _fresh_collision(self) -> float:
        """Re-settle every stored situation, then take the largest |cosine| between answers that
        share a present and differ in context.

        This, and not a comparison of stored answers, is what the verdict needs: the stored codes
        belong to the graph as it was, so a comparison against them cannot see the growth at all
        and every attempt would report an unchanged collision.
        """
        worst = 0.0
        by_present: dict[int, list[np.ndarray]] = {}
        for (present, _context), drive in self._drives.items():
            # a stored situation outlives a growth event, so its drive must be widened to the
            # current neuron count; appended neurons are driven by nothing the world declared.
            if drive.shape[1] != self.neurons:
                wide = np.zeros((drive.shape[0], self.neurons))
                wide[:, : drive.shape[1]] = drive
                drive = wide
            code, _ = self._settle(drive)
            by_present.setdefault(present, []).append(code)
        for codes in by_present.values():
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    a, b = codes[i], codes[j]
                    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                    if na and nb:
                        worst = max(worst, abs(float(a @ b) / (na * nb)))
        return worst

    # -- 3: the generator ------------------------------------------------
    def _neighbours(self, seeds: np.ndarray) -> np.ndarray:
        pre, post = self.connectome.pre, self.connectome.post
        mask = np.isin(pre, seeds) | np.isin(post, seeds)
        return np.unique(np.concatenate([pre[mask], post[mask]]))

    def _difference_cells(self, confused: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
        """The cells that carry what the colliding situations differ in.

        Paired inside one present input: same sense, different context. With
        ``difference_source = "state"`` the comparison runs over the whole state, which is where
        the distinction usually lives, and the returned mask is over all neurons; with
        ``"readout"`` it runs over the declared answer only, which is what the stored codes hold
        and is the arm this paper measured against. Reading the difference out of the answer is
        self-defeating whenever the answer is nearly proportional: the difference vector is then a
        small multiple of the answer itself, and wiring from it copies the answer instead of the
        distinction.
        """
        if self.config.difference_source == "state" and self._full:
            by_present: dict[int, list[np.ndarray]] = {}
            for _code, present, context in confused:
                state = self._full.get((present, context))
                if state is not None:
                    by_present.setdefault(present, []).append(state)
            diffs = [np.abs(a - b) for states in by_present.values()
                     for a, b in zip(states, states[1:])]
            if diffs:
                peak = np.stack(diffs).max(axis=0)
                cut = np.quantile(peak[peak > 0], 1.0 - self.config.difference_fraction) if (peak > 0).any() else 1.0
                return peak >= max(cut, 1e-9)
            return np.zeros(self.neurons, dtype=bool)
        by_present = {}
        for code, present, _ in confused:
            by_present.setdefault(present, []).append(code)
        diffs = [np.abs(a - b) for codes in by_present.values() for a, b in zip(codes, codes[1:])]
        if not diffs:
            return np.zeros(self.neurons, dtype=bool)
        mask = np.zeros(self.neurons, dtype=bool)
        mask[self.readout] = np.stack(diffs).max(axis=0) > 1e-6
        return mask

    def _grow(self, confused: list[tuple[np.ndarray, int, int]], rng: np.random.Generator) -> int:
        cfg = self.config
        n_old, k = self.neurons, cfg.neurons_per_event
        if n_old + k > cfg.max_neurons:
            return 0
        # what to wire: the readout cells the confusion lives in, their sources,
        # and the cells that are active while the answer forms (one hop out).
        hot = np.zeros(n_old, dtype=bool)
        differs = self._difference_cells(confused)
        hot[: len(differs)] = differs            # a mask over all neurons, or over the readout
        for code, _, _ in confused:
            hot[self.readout[np.argsort(-code)[:4]]] = True
        state = self.brain.settle_batch(self._drive(("odour_A", 1.0)), steps=self.chunk, state=None)
        hot |= np.asarray(state.activation).ravel() > 0.2
        seeds = np.flatnonzero(hot)
        near = self._neighbours(seeds) if len(seeds) else np.arange(n_old)
        if len(near) < 4:
            near = np.arange(n_old)
        # the cells that carry the distinction are the pool a new neuron should read first:
        # a neuron fed the same activity in both situations cannot tell them apart, however
        # many edges it has.
        diff_idx = np.flatnonzero(differs)[:64]
        near = np.union1d(near, diff_idx) if len(diff_idx) else near
        mean_in = max(1.0, self.synapses / n_old)
        init = cfg.init_scale * (float(np.mean(np.abs(self._efficacy))) or 0.02)

        pre, post, sign = [], [], []
        for j in range(k):
            new = n_old + j
            for which in (0, 1):  # in-edges, then out-edges, one degree prior
                degree = int(np.clip(rng.lognormal(np.log(cfg.degree), cfg.degree_jitter),
                                     1, 8 * mean_in))
                if which == 0 and len(diff_idx):
                    from_diff = rng.random(degree) < cfg.difference_share
                    partners = np.where(
                        from_diff,
                        diff_idx[rng.integers(0, len(diff_idx), degree)],
                        near[rng.integers(0, len(near), degree)],
                    )
                else:
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
                # excitation cannot move a saturated cell; inhibition can move it out of
                # saturation, which is the only way an added synapse can separate two answers
                # that both sit at the ceiling.
                target = int(self.readout[rng.integers(0, len(self.readout))])
                s = cfg.readout_sign
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
        before = self._fresh_collision()
        prev_connectome, prev_efficacy = self.connectome, self._efficacy.copy()
        t0 = time.perf_counter()
        added = self._grow(confused, np.random.default_rng(self.ticks + 7))
        grow_ms = 1000 * (time.perf_counter() - t0)
        if not added:
            return {"event": "budget", "before": before, "after": before, "added": 0}
        after = self._fresh_collision()
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
        drive = self._drive(present, context)
        code, full, info = self._settle_full(drive)
        if float(np.linalg.norm(code)) == 0.0:
            self.silent += 1  # a silent answer is a response failure, not a capacity one
        collision = self._collision(code, present_id, context_id)
        self._remember(code, present_id, context_id, drive, full)
        # the prerequisite is a property of the pair just seen, so it is recomputed per moment
        self.state_motion = self._state_motion()
        if self.state_motion < self.config.min_state_motion:
            # the premise of growth is false here: the declared pairs do not move the
            # answer, so no amount of capacity can separate them. Say so, do not grow.
            self._evidence = 0
            self.log.append({"event": "no_response", "state_motion": round(self.state_motion, 8),
                             "pair_motion": round(self.pair_motion, 5),
                             "spread": round(self.readout_spread, 4), "neurons": self.neurons})
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
          f"per-sweep {per_sweep:6.1f} us | state motion {ep.state_motion:.2e} | "
          f"{ms_tick:5.1f} ms/tick | "
          f"kept {ep.growths_kept:>2d} | failed {ep.growths_failed:>2d}")


def demo(ticks: int = 1200, readout: tuple[str, ...] = ("forward", "reverse"),
         source: str = "state", substrate: str = "worm", init_scale: float = 3.0,
         readout_sign: float = 1.0, drive_scale: float = 1.0,
         hetero_spread: float = 0.0) -> None:
    """Start at 302 neurons; the world keeps requiring new distinctions.

    ``readout`` is the declared answer. The locomotion readout fails the prerequisite on
    this substrate; the sensory readout contains the cells that carry the context, so the
    pairs do move and the growth layer is exercised rather than refused.
    """
    context = "context_tone" if substrate == "tone" else "pain"
    ep = EP4NN(load_worm(tone=substrate == "tone"),
               GrowthConfig(readout=readout, difference_source=source, context=context,
                            init_scale=init_scale, readout_sign=readout_sign,
                            drive_scale=drive_scale, hetero_spread=hetero_spread))
    print(f"start: neurons={ep.neurons} synapses={ep.synapses} "
          f"(row-normalised operating point {ep.config.row_sum})")
    rng = np.random.default_rng(3)
    contexts = 2                                   # pain levels that must be told apart
    t0 = time.perf_counter()
    for i in range(ticks):
        if i and i % 240 == 0:
            contexts += 1
            _report("milestone", ep, t0, i)
            print(f"           tick {i}: the world now requires {contexts} levels of "
                  f"{ep.config.context} to be told apart")
            t0 = time.perf_counter()
        context_id = int(rng.integers(0, contexts))
        level = 0.35 + 0.65 * context_id / max(1, contexts - 1)
        out = ep.tick(("odour_A", 1.0), (ep.config.context, level), 0, context_id)
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
                 "state_motion": round(ep.state_motion, 8),
                 "min_state_motion": ep.config.min_state_motion,
                 "pair_motion": round(ep.pair_motion, 5),
                 "difference_source": ep.config.difference_source,
                 "readout": list(ep.config.readout),
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
    which = sys.argv[1] if len(sys.argv) > 1 else "default"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    source = sys.argv[3] if len(sys.argv) > 3 else "state"
    ticks = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    rs = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    drive = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    hetero = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    tone = which == "tone"
    demo(readout={"default": ("forward", "reverse"), "sensory": ("class_sensory",),
                  "interneuron": ("class_interneuron",), "tone": ("forward", "reverse")}
         .get(which, ("forward", "reverse")),
         source=source, substrate="tone" if tone else "worm", init_scale=scale, ticks=ticks,
         readout_sign=rs, drive_scale=drive, hetero_spread=hetero)
