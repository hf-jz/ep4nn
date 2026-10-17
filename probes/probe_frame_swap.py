"""Same task, same frame, different substrate: cadence's composed brain against the worm graph.

The comparison the small examples cannot give: one non-trivial continuing task, solved twice by the
library's own application frame (`cadence.generic.Brain`: basal ganglia, arousal, working memory), with
the substrate swapped. The task is the beat from `examples/walking_for_reward.py` — one constant drive,
reward one unit whenever the action differs from the previous one, nothing in the observation saying
which action came last. Efference is off for every arm so the substrates are compared on the same
information.

    python probes/probe_frame_swap.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ep4nn.grow import EP4NN, GrowthConfig, load_worm   # noqa: E402
from dataclasses import replace                        # noqa: E402
from cadence import ArousalConfig, Brain as ComposedBrain   # noqa: E402
from cadence import Connectome                         # noqa: E402
from cadence.generic import Brain as FrameBrain        # noqa: E402

OUT: dict[str, object] = {}
DRIVE = np.array([[1.0, 0.0, 0.0, 0.0]])
MOMENTS = 600          # the example's own length
TAIL = 100             # after its 100-moment youth
AROUSAL = {"threshold": 0.2, "decay": 0.9, "tolerance": 2.0, "floor": 0.1, "fast": 0.05,
           "slow": 0.005, "heat": 2.0, "youth": 100, "value_surprise": 1.0,
           "record_surprise": 0.0, "need": 0.5}


def worm_with_ports(grown: int = 0) -> Connectome:
    """The measured wiring, renamed so the application frame can drive and read it."""
    conn = load_worm()
    pops = dict(conn.populations)
    pops["sensory"] = pops["class_sensory"][:4]          # the four cells the drive reaches
    pops["association"] = pops["class_interneuron"]
    pops["motor"] = pops["class_command"]                # the action pool
    if grown:
        ep = EP4NN(conn, GrowthConfig(calibrate=False, degree=4.5, to_readout=0.4, init_scale=1.0))
        codes = list(ep._situation_codes())
        events = 0
        while ep.neurons < grown and events < 200:
            ep.ticks = events
            if not ep._grow([(codes[0], 0, 0), (codes[-1], 0, len(codes) - 1)],
                            np.random.default_rng(200 + events)):
                break
            ep._rebuild()
            events += 1
        wide = ep.connectome
    else:
        wide = Connectome.from_synapses(conn.n, pre=conn.pre, post=conn.post, count=conn.count,
                                        sign=conn.sign, populations=pops, label=conn.label)
    wide_pops = {**wide.populations, **{k: pops[k] for k in ("sensory", "association", "motor")}}
    return Connectome.from_synapses(wide.n, pre=wide.pre, post=wide.post, count=wide.count,
                                    sign=wide.sign, populations=wide_pops, label=wide.label)


def beat(brain, moments: int = MOMENTS) -> dict:
    previous, reward, earned = None, None, []
    t0 = time.perf_counter()
    actions = []
    for _ in range(moments):
        feedback = ({} if reward is None else
                    {"reward": np.array([reward]), "done": np.array([False])})
        action = int(brain.live(DRIVE, **feedback)[0])
        reward = 0.0 if previous is None else float(action != previous)
        previous = action
        actions.append(action)
        earned.append(reward)
    elapsed = time.perf_counter() - t0
    tail = earned[-TAIL:]
    return {"feeding_tail": round(float(np.mean(tail)), 4),
            "feeding_all": round(float(np.mean(earned)), 4),
            "distinct_actions": len(set(actions)),
            "ms_per_moment": round(1000 * elapsed / moments, 2),
            "neurons": int(brain.connectome.n), "synapses": int(brain.connectome.synapses)}


def main() -> None:
    arms = {}

    def composed_brain(copy: bool):
        """The example's founder, at its declared point: module size, arousal genes, actor."""
        # compose already returns the application frame; `.brain` is the raw equilibrium brain
        brain = ComposedBrain.compose(4, 2, modules=(32,), seed=3,
                                      arousal=ArousalConfig(**AROUSAL),
                                      efference_amplitude=3.0 if copy else 0.0,
                                      efference_decay=0.0)
        brain.basal_ganglia.config = replace(brain.basal_ganglia.config, eta=0.1, eta_bias=0.01,
                                             lam=0.95, gamma=0.95, eta_critic=5.0)
        return brain

    arms["cadence compose, declared point, efference on"] = composed_brain(True)
    arms["cadence compose, declared point, no efference"] = composed_brain(False)
    for label, grown in (("worm 302 (ep4nn substrate)", 0), ("worm grown to 606 (ep4nn growth)", 606)):
        arms[label] = FrameBrain(worm_with_ports(grown), slots=1,
                                 arousal=ArousalConfig(**AROUSAL), seed=3,
                                 efference_amplitude=0.0)
    rows = []
    for label, brain in arms.items():
        row = {"arm": label, **beat(brain)}
        rows.append(row)
        print(f"{label:<38} neurons {row['neurons']:>5} synapses {row['synapses']:>7} "
              f"feeding(tail) {row['feeding_tail']:.3f} actions {row['distinct_actions']} "
              f"{row['ms_per_moment']:>7.1f} ms/moment")
    OUT["rows"] = rows
    OUT["task"] = {"moments": MOMENTS, "tail": TAIL, "drive": DRIVE.tolist(),
                   "reward": "one unit when the action differs from the previous one"}
    dump = Path(__file__).resolve().parents[1] / "docs" / "assets" / "frame_swap.json"
    dump.write_text(json.dumps(OUT, indent=1) + "\n")
    print(f"wrote {dump}")


if __name__ == "__main__":
    main()
