"""Probe 9: does adaptation turn the same 302-neuron wiring into a rhythm?

The library's own neuron model says adaptation is what lets a connectome with
mutual inhibition produce rhythm instead of a fixed point, while PatchNet's
learning contract requires an adaptation-free (fixed-point) structure.
Measured here: settle the same drive with and without adaptation, and report the
residual and the movement of the activity after many steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cadence import Connectome
from cadence.brain import Brain
from cadence.learning import learning_neuron_model
from cadence.neuron import Adaptation, NeuronModel

OUT: dict[str, object] = {}


def build() -> Connectome:
    raw = json.loads(Path("/Users/sheng/math/ep4nn/data/worm_connectome.json").read_text())
    pre, post, count, sign = [], [], [], []
    for a, b, c, s in ((int(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in raw["chemical"]):
        if a != b:
            pre.append(a); post.append(b); count.append(c); sign.append(s)
    for a, b, c in ((int(r[0]), int(r[1]), float(r[2])) for r in raw["gap"]):
        if a != b:
            for x, y in ((a, b), (b, a)):
                pre.append(x); post.append(y); count.append(c); sign.append(1.0)
    return Connectome.from_synapses(len(raw["neurons"]), pre=pre, post=post, count=count,
                                    sign=sign, label="celegans:c302_A_Full")


def probe(conn: Connectome, model, label: str, steps: int = 1500) -> dict:
    brain = Brain(conn, model, backend="cpu")
    drive = np.zeros((1, conn.n))
    drive[0, :4] = 1.0
    state = None
    traj = []
    for block in range(steps // 100):
        state = brain.settle_batch(drive, steps=100, state=state)
        traj.append(np.asarray(state.activation).ravel().copy())
    traj = np.array(traj)  # (blocks, neurons)
    last = traj[-1]
    earlier = traj[-2]
    movement = float(np.max(np.abs(last - earlier)))
    residual = float(np.max(brain.residual(drive, state)))
    # a limit cycle shows as sustained movement of the same size, a fixed point as zero
    sizes = [float(np.max(np.abs(traj[i + 1] - traj[i]))) for i in range(len(traj) - 1)]
    return {
        "label": label,
        "residual_after_1500_steps": residual,
        "movement_last_100_steps": movement,
        "movement_per_100_steps": [round(s, 6) for s in sizes[-6:]],
        "active_neurons": int((last > 1e-9).sum()),
        "peak_activity_last_block": float(last.max()),
        "activity_never_constant": bool(movement > 1e-6),
    }


def main() -> None:
    conn = build()
    base = learning_neuron_model()
    OUT["no_adaptation"] = probe(conn, base, "learning_neuron_model (adaptation off)")
    for tau, strength in ((50.0, 1.0), (20.0, 2.0), (100.0, 0.5)):
        model = NeuronModel(
            dt=base.dt, slope=base.slope, threshold=base.threshold, gain=base.gain,
            stimulus_amplitude=base.stimulus_amplitude, leak=base.leak,
            adaptation=Adaptation(tau_steps=tau, strength=strength),
        )
        OUT[f"adaptation_tau{tau}_s{strength}"] = probe(conn, model, f"adaptation tau={tau} strength={strength}")
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
