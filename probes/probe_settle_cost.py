"""Probe 1: what a continuing-life tick costs in the cadence library (local checkout)."""

from __future__ import annotations

import json
import time

import numpy as np

from cadence import Brain

OUT: dict[str, object] = {}


def facts(b: Brain) -> dict:
    core = b.brain
    nm = core.neuron_model
    w = core.weights
    absrow = np.zeros(core.connectome.n)
    np.add.at(absrow, core.connectome.post, np.abs(w))
    lip = getattr(nm, "lipschitz", None)
    return {
        "neurons": int(core.connectome.n),
        "synapses": int(core.connectome.synapses),
        "populations": {k: len(v) for k, v in core.connectome.populations.items()},
        "sensory": len(b.sensory_index),
        "motor": len(b.motor_index),
        "neuron_model": type(nm).__name__,
        "gain": float(nm.gain),
        "dt": float(nm.dt),
        "activation": getattr(nm, "activation", None).__name__,
        "adaptation": None if getattr(nm, "adaptation", None) is None else "on",
        "max_row_sum_abs_w": float(absrow.max()),
        "mean_row_sum_abs_w": float(absrow.mean()),
        "contraction_q_if_L1": float(1 - nm.dt + nm.dt * float(absrow.max())),
    }


def live_run(b: Brain, ticks: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    n_in = len(b.sensory_index)
    obs = np.eye(n_in)
    sweeps, checks, residuals, times, qualified = [], [], [], [], 0
    b.live(obs[[0]])
    for _ in range(ticks):
        x = obs[rng.integers(0, n_in)][None, :]
        r = float(rng.random() < 0.4)
        t0 = time.perf_counter()
        b.live(x, reward=np.array([r]))
        times.append(time.perf_counter() - t0)
        s = b.last_settlement
        sweeps.append(int(s["steps"]))
        checks.append(int(s["residual_checks"]))
        residuals.append(float(s["max_residual"]))
        qualified += int(bool(s["qualified"]))
    return {
        "ticks": ticks,
        "sweeps_mean": round(float(np.mean(sweeps)), 2),
        "sweeps_median": float(np.median(sweeps)),
        "sweeps_max": int(np.max(sweeps)),
        "sweeps_sum": int(np.sum(sweeps)),
        "residual_checks_sum": int(np.sum(checks)),
        "residual_last": residuals[-1],
        "residual_worst": float(np.max(residuals)),
        "qualified_all": bool(qualified == ticks),
        "ms_per_tick_mean": round(1000 * float(np.mean(times)), 3),
        "ms_per_tick_median": round(1000 * float(np.median(times)), 3),
        "ticks_per_second": round(float(1.0 / np.mean(times)), 1),
    }


def main() -> None:
    for modules in [(16, 8), (64, 32, 16), (256, 128, 64)]:
        b = Brain.compose(inputs=8, actions=4, modules=modules, arousal=True, seed=3)
        OUT[f"facts{modules}"] = facts(b)
        OUT[f"live{modules}"] = live_run(b, 60)

    b = Brain.compose(inputs=8, actions=4, modules=(64, 32, 16), arousal=True, seed=3)
    core = b.brain
    drive = np.zeros((1, core.connectome.n))
    drive[0, list(b.sensory_index)] = 1.0
    trials = {}
    for tol in (1e-6, 1e-5, 1e-4, 1e-3):
        t0 = time.perf_counter()
        cold = core.equilibrate(drive, budget=2048, chunk=32, tolerance=tol)
        t_cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        warm = core.equilibrate(drive, budget=2048, chunk=32, tolerance=tol, state=cold.state)
        t_warm = time.perf_counter() - t0
        moved = (np.abs(warm.state.activation - cold.state.activation)).max()
        trials[f"tol={tol:g}"] = {
            "cold_sweeps": int(cold.steps),
            "cold_checks": int(cold.residual_checks),
            "cold_residual": float(np.max(cold.residual)),
            "cold_ms": round(1000 * t_cold, 3),
            "warm_same_input_sweeps": int(warm.steps),
            "warm_ms": round(1000 * t_warm, 3),
            "warm_state_move": float(moved),
        }
    OUT["tolerance_sweep"] = trials

    # cost split: one transport vs one full sweep of the settle loop
    core2 = b.brain
    s = np.zeros((1, core2.connectome.n)) + 0.1
    t0 = time.perf_counter()
    for _ in range(200):
        core2._synaptic_input(s)
    t_transport = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        core2.settle_batch(drive, steps=1, state=None)
    t_step = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        core2.residual(drive, cold.state)
    t_resid = (time.perf_counter() - t0) / 200
    OUT["per_op_us"] = {
        "transport": round(1e6 * t_transport, 1),
        "one_sweep_settle_batch": round(1e6 * t_step, 1),
        "residual": round(1e6 * t_resid, 1),
    }
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
