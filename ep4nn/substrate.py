"""One substrate contract for any connectome: the graph is data, the ports are the ABI.

The worm and the fly differ in wiring, not in kind: both are (n, pre, post, count, sign) plus a
vocabulary of cell classes. What the library's application frame needs is not the species, it is a
handful of named populations to drive (`sensory`), to read (`motor`) and, when the task is a
continuing life, to copy (`efference`). So the portable piece is a *population spec*, and a substrate
is a connectome that can answer for one.

`Substrate.require()` is the honest half: a frame asked for a port the graph cannot supply must fail
with the name of the port, not quietly run a brain that can never learn. The self-check at the bottom
runs two architectures through the same adapter and shows exactly that.

    python -m ep4nn.substrate
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from .grow import load_worm
except ImportError:                                            # run as a script
    from grow import load_worm

from cadence import Connectome

# What `cadence.generic.Brain` looks for. Anything else in a spec is for our own layering.
FRAME_PORTS = ("sensory", "association", "motor", "efference")


@dataclass(frozen=True)
class PopulationSpec:
    """The port ABI: names in a connectome's own vocabulary, one per frame port.

    `required` is what the task needs, `available` is what the graph declares. Keeping the two apart
    is the whole point: a species that lacks efference can still be measured, it cannot be a
    continuing life.
    """

    sensory: str
    association: str
    motor: str
    efference: str | None = None
    readout: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def ports(self) -> dict[str, str]:
        out = {"sensory": self.sensory, "association": self.association, "motor": self.motor}
        if self.efference:
            out["efference"] = self.efference
        return out


class Substrate:
    """A connectome plus the names that let a frame drive it. No species-specific code below."""

    def __init__(self, connectome: Connectome, spec: PopulationSpec, label: str = ""):
        self.connectome = connectome
        self.spec = spec
        self.label = label or connectome.label

    # -- the port contract -------------------------------------------------------------------
    def missing(self, needed: tuple[str, ...] = FRAME_PORTS) -> list[str]:
        present = set(self.connectome.populations)
        have = {port for port, name in self.spec.ports().items() if name in present}
        return [port for port in needed if port not in have]

    def require(self, needed: tuple[str, ...] = FRAME_PORTS) -> "Substrate":
        gap = self.missing(needed)
        if gap:
            raise LookupError(f"substrate {self.label!r} cannot supply {gap}; "
                              f"it declares {sorted(self.connectome.populations)[:12]}")
        return self

    # -- the library's view ------------------------------------------------------------------
    def library_view(self) -> Connectome:
        """The same graph, renamed so `cadence.generic.Brain` finds its ports."""
        pops = dict(self.connectome.populations)
        for port, name in self.spec.ports().items():
            pops[port] = pops[name]
        return Connectome.from_synapses(self.connectome.n, pre=self.connectome.pre,
                                        post=self.connectome.post, count=self.connectome.count,
                                        sign=self.connectome.sign, populations=pops,
                                        label=self.label)

    def frame(self, **kwargs):
        from cadence.generic import Brain as FrameBrain

        return FrameBrain(self.library_view(), **kwargs)

    # -- the ep4nn-side metric ---------------------------------------------------------------
    def measure(self, situations: int = 9, seed: int = 0, steps: int = 64) -> dict:
        """Spread and collision of the readout across situations, plus the cost of settling.

        A substrate-level number: it says whether this graph can tell the situations apart at all,
        before any task, frame or growth is involved.
        """
        from cadence.brain import Brain
        from cadence.learning import learning_neuron_model

        brain = Brain(self.connectome, learning_neuron_model(), backend="cpu")
        cells = np.asarray(self.connectome.populations[self.spec.sensory]).ravel()
        out = np.flatnonzero(np.asarray(
            self.connectome.populations[self.spec.readout or self.spec.motor]).ravel())
        rng = np.random.default_rng(seed)
        answers = []
        t0 = time.perf_counter()
        for _ in range(situations):
            drive = np.zeros(self.connectome.n)
            level = rng.uniform(0.5, 1.0, size=len(cells))
            drive[cells] = level
            eq = brain.equilibrate(drive, budget=1024, chunk=steps, tolerance=1e-4)
            answers.append(np.asarray(eq.state.activation).ravel()[out])
        per_settle = 1000 * (time.perf_counter() - t0) / situations
        codes = np.array(answers)
        norms = np.linalg.norm(codes, axis=1)
        coll = 1.0 if norms.min() == 0 else max(
            abs(float(codes[i] @ codes[j]) / (norms[i] * norms[j]))
            for i in range(situations) for j in range(i + 1, situations))
        return {"neurons": self.connectome.n, "synapses": self.connectome.synapses,
                "spread": round(float(np.mean(codes.max(axis=0) - codes.min(axis=0))), 6),
                "collision": round(float(coll), 6), "settle_ms": round(per_settle, 2)}


def worm_substrate(grown: int = 0) -> Substrate:
    conn = load_worm()
    spec = PopulationSpec(sensory="class_sensory", association="class_interneuron",
                          motor="class_command")
    if not grown:
        return Substrate(conn, spec, label=f"worm {conn.n}")
    from .grow import EP4NN, GrowthConfig

    ep = EP4NN(conn, GrowthConfig(calibrate=False, degree=4.5, to_readout=0.4, init_scale=1.0))
    codes = list(ep._situation_codes())
    events = 0
    while ep.neurons < grown and events < 200:
        ep.ticks = events
        if not ep._grow([(codes[0], 0, 0), (codes[-1], 0, len(codes) - 1)],
                        np.random.default_rng(300 + events)):
            break
        ep._rebuild()
        events += 1
    return Substrate(ep.connectome, spec, label=f"worm grown {ep.connectome.n}")


def layered_substrate(n_region: int = 6000, seed: int = 0) -> Substrate:
    """A three-region architecture the worm does not have: input sheet, recurrent centre, descending
    cells. Nothing about it is species-specific; it is here to prove the adapter is not worm-shaped.
    """
    rng = np.random.default_rng(seed)
    optic = np.arange(n_region)
    centre = np.arange(n_region, 3 * n_region)
    descending = np.arange(3 * n_region, 3 * n_region + 32)
    efference = np.arange(3 * n_region + 32, 3 * n_region + 64)
    n = 3 * n_region + 64
    pre, post = [], []
    for source, target, degree in ((optic, centre, 6), (centre, centre, 8),
                                   (centre, descending, 12), (efference, centre, 4)):
        chosen = rng.integers(0, len(target), size=(len(source), degree))
        pre.append(np.repeat(source, degree))
        post.append(np.asarray(target)[chosen].ravel())
    pre = np.concatenate(pre)
    post = np.concatenate(post)
    return Substrate(Connectome.from_synapses(n, pre=pre, post=post,
                                              populations={"sensory": optic,
                                                           "association": centre,
                                                           "motor": descending,
                                                           "efference": efference},
                                              label=f"layered {n}"),
                     PopulationSpec(sensory="sensory", association="association", motor="motor",
                                    efference="efference", readout="motor"),
                     label="layered optic/centre/descending")


def demo() -> None:
    """Two architectures, one adapter. The worm reports its missing port instead of pretending."""
    for build in (worm_substrate, layered_substrate):
        sub = build()
        gap = sub.missing()
        print(f"{sub.label:<34} neurons {sub.connectome.n:>6} synapses {sub.connectome.synapses:>7} "
              f"missing ports {gap or 'none'}")
        print(f"{'':<34} {sub.measure()}")
        try:
            sub.frame(slots=1, arousal=True, seed=3, efference_amplitude=0.0)
            framed = "frame built"
        except Exception as exc:                                # the honest branch
            framed = f"frame refused: {type(exc).__name__}: {exc}"
        print(f"{'':<34} {framed}")
        assert sub.measure()["collision"] <= 1.0
    assert "efference" in worm_substrate().missing(), "the worm graph grew an efference sheet"
    assert layered_substrate().missing() == [], "the layered graph cannot answer for its own ports"


if __name__ == "__main__":
    demo()
