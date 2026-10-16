"""Per-neuron heterogeneous rate dynamics, without touching the library.

Why this exists: the paper lists three remedies for the prerequisite failure and this is one of
them. In a measured connectome every cell gets the same slope and threshold, so the readout's
answers end up parallel; real cells do not share those constants. The library's model carries them
as scalars, so this is a duck-typed model with arrays for slope and threshold and the same surface
the library reads: ``dt``, ``gain``, ``stimulus_amplitude``, ``leak``, ``adaptation``,
``rest_emission``, ``threshold``, ``activation``, ``slope_at``, ``replace``, ``_validate_precision``,
``to_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HeteroRateModel:
    """Rectified sigmoid with one slope and one threshold per neuron.

    ``spread`` is the log-normal width of the per-neuron constants: 0 reproduces the library's
    scalar model exactly, 0.5 makes the slowest cell's slope about a third of the fastest's.
    """

    dt: float = 0.5
    slope: Any = 4.0
    threshold: Any = 1.5
    gain: float = 0.02
    stimulus_amplitude: float = 3.0
    leak: float = 0.0
    adaptation: None = None
    spread: float = 0.5
    seed: int = 0
    n: int = 0

    def __post_init__(self) -> None:
        n = self.n or (len(self.slope) if hasattr(self.slope, "__len__") else 0)
        if n and not hasattr(self.slope, "__len__"):
            rng = np.random.default_rng(self.seed)
            z = rng.standard_normal(n)
            jitter = np.exp(self.spread * z - 0.5 * self.spread ** 2)   # mean stays at the base value
            base_s, base_t = float(self.slope), float(self.threshold)
            # keep both constants in a valid range: a jittered threshold may not cross zero, or the
            # resting sigmoid saturates and the reciprocal in the activation blows up
            slope = np.clip(np.full(n, base_s) * jitter, 0.25 * base_s, 4.0 * base_s)
            threshold = np.clip(np.full(n, base_t) * (2.0 - jitter), 0.4 * base_t, 2.5 * base_t)
            object.__setattr__(self, "slope", slope)
            object.__setattr__(self, "threshold", threshold)
            object.__setattr__(self, "n", n)

    @property
    def rest_emission(self) -> Any:
        s = np.exp(np.asarray(self.slope) * np.asarray(self.threshold))
        return 1.0 / (1.0 + s)

    def activation(self, v: np.ndarray) -> np.ndarray:
        rest = np.minimum(self.rest_emission, 1.0 - 1e-12)
        r = np.exp(-np.asarray(self.slope) * (v - np.asarray(self.threshold)))
        r = 1.0 / (1.0 + r)
        r = r - rest
        if self.leak == 0.0:
            r = np.maximum(r, 0.0)
            return r * (1.0 / (1.0 - rest))
        return np.where(r > 0.0, r * (1.0 / (1.0 - rest)), r * (self.leak / rest))

    def slope_at(self, v: np.ndarray) -> np.ndarray:
        rest = np.minimum(self.rest_emission, 1.0 - 1e-12)
        sig = 1.0 / (1.0 + np.exp(-np.asarray(self.slope) * (v - np.asarray(self.threshold))))
        d = np.asarray(self.slope) * sig * (1.0 - sig)
        return np.where(sig > rest, d / (1.0 - rest), self.leak * d / rest)

    def _validate_precision(self, dtype: str) -> None:
        rest = self.rest_emission
        values = np.concatenate([np.atleast_1d(np.asarray(x, float)) for x in
                                 (self.dt, self.gain, self.stimulus_amplitude, rest)])
        if not np.isfinite(np.asarray(values, dtype=dtype)).all():
            raise ValueError("neuron rule is not representable at runtime precision")

    def replace(self, **changes: Any) -> "HeteroRateModel":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {"model": "hetero rate", "dt": self.dt, "gain": self.gain,
                "stimulus_amplitude": self.stimulus_amplitude, "leak": self.leak,
                "spread": self.spread, "seed": self.seed, "n": self.n}


def _self_check() -> None:
    from cadence import Connectome
    from cadence.brain import Brain
    import json
    from pathlib import Path

    raw = json.loads((Path(__file__).resolve().parents[1] / "data" / "worm_connectome.json").read_text())
    pre, post, count, sign = [], [], [], []
    for a, b, c, s in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
        if a != b:
            pre.append(a); post.append(b); count.append(c); sign.append(s)
    for a, b, c in ((int(r[0]), int(r[1]), float(r[2])) for r in raw["gap"]):
        if a != b:
            for x, y in ((a, b), (b, a)):
                pre.append(x); post.append(y); count.append(c); sign.append(1.0)
    n = len(raw["neurons"])
    conn = Connectome.from_synapses(n, pre=pre, post=post, count=count, sign=sign, label="worm")
    indeg = np.maximum(np.bincount(conn.post, minlength=n), 1)
    eff = conn.sign * (0.5 / indeg[conn.post])
    model = HeteroRateModel(n=n, spread=0.5, seed=3)
    assert model.slope.shape == (n,) and model.threshold.shape == (n,)
    assert abs(float(np.mean(model.slope)) - 4.0) < 0.2, float(np.mean(model.slope))
    assert model.slope.max() / model.slope.min() > 2.0, float(model.slope.max() / model.slope.min())
    brain = Brain(conn, model, backend="cpu", efficacy=eff)
    drive = np.zeros((1, n)); drive[0, :8] = 1.0
    eq = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-4)
    act = np.asarray(eq.state.activation).ravel()
    assert np.isfinite(act).all() and act.max() > 0.05, (float(act.max()), float(act.min()))
    print(f"slope {model.slope.min():.2f}-{model.slope.max():.2f}, "
          f"threshold {model.threshold.min():.2f}-{model.threshold.max():.2f}")
    print(f"hetero self-check ok: settled in {int(eq.steps)} sweeps, "
          f"active {int((act > 1e-9).sum())}/{n}, max {act.max():.4f}")


if __name__ == "__main__":
    _self_check()
