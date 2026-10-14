"""Probe 7: can one continuing Cadence brain hold a moving setpoint on a plant?

Plant: m x'' = u - k x - c x' (k=2, c=1, m=1, underdamped; the setpoint is +-1).
Observation: [error, error rate, reference, 1].
Action: one of five force levels (-4, -2, 0, +2, +4): enough authority for x ~ +-1.
Reward: 1 while |error| < 0.25, else 0, supplied on the next call as the
documented measured outcome of the preceding action.
Arms: uniform-random, frozen greedy brain, learning brain via `live` (arousal),
learning brain via `step` (learns from every outcome).
"""

from __future__ import annotations

import json

import numpy as np

from cadence import Brain

LEVELS = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
K, C, M, DT = 2.0, 1.0, 1.0, 0.02
TARGET_BAND = 0.25


def plant_step(x, v, u):
    a = (u - K * x - C * v) / M
    v += a * DT
    x += v * DT
    return x, v


def reference(t, period=4.0):
    return 1.0 if int(t / period) % 2 == 0 else -1.0


def observe(r, x, v):
    return np.array([[np.clip(r - x, -2, 2), np.clip(v, -3, 3), r, 1.0]])


def run(ticks: int, brain: Brain | None, mode: str, seed: int = 1, dense: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    x = v = t = 0.0
    errors, rewards, chosen = [], [], []
    action = None
    reward_prev = np.array([0.0])
    for _ in range(ticks):
        t += DT
        r = reference(t)
        obs = observe(r, x, v)
        if mode == "random":
            action = int(rng.integers(0, len(LEVELS)))
        elif action is None:
            action = int((brain.act(obs) if mode == "step" else brain.live(obs))[0])
        elif mode == "step":
            action = int(brain.step(obs, reward=reward_prev)[0])
        else:
            action = int(brain.live(obs, reward=reward_prev)[0])
        x, v = plant_step(x, v, LEVELS[action])
        err = abs(r - x)
        reward_prev = np.array(
            [1.0 - min(1.0, err) if dense else (1.0 if err < TARGET_BAND else 0.0)]
        )
        errors.append(err)
        rewards.append(1.0 if err < TARGET_BAND else 0.0)
        chosen.append(action)
    q = 3 * ticks // 4
    hist = np.bincount(chosen, minlength=len(LEVELS)) / len(chosen)
    return {
        "mode": mode + ("/dense" if dense else "/sparse"),
        "mean_abs_error": round(float(np.mean(errors)), 4),
        "mean_abs_error_last_quarter": round(float(np.mean(errors[q:])), 4),
        "in_target_band_all": round(float(np.mean(rewards)), 4),
        "in_target_band_last_quarter": round(float(np.mean(rewards[q:])), 4),
        "action_histogram_last_quarter": [round(float(h), 3) for h in
                                          np.bincount(chosen[q:], minlength=len(LEVELS))
                                          / len(chosen[q:])],
    }


def fresh(seed=5) -> Brain:
    brain = Brain.compose(inputs=4, actions=5, modules=(64, 32, 16), arousal=True, seed=seed)
    brain.live(np.zeros((1, 4)))
    return brain


def frozen(seed=5) -> Brain:
    brain = fresh(seed)
    brain.retune(arousal_youth=0, arousal_threshold=50.0, arousal_decay=0.0,
                 arousal_floor=0.0, learning_free_steps=1024)
    return brain


def main() -> None:
    ticks = 6000
    out = {}
    out["random"] = run(ticks, None, "random", seed=1)
    out["frozen_greedy_sparse"] = run(ticks, frozen(), "live", seed=1, dense=False)
    out["live_arousal_sparse"] = run(ticks, fresh(), "live", seed=1, dense=False)
    out["live_arousal_dense"] = run(ticks, fresh(), "live", seed=1, dense=True)
    out["live_arousal_dense_seed2"] = run(ticks, fresh(seed=11), "live", seed=1, dense=True)

    b_step = fresh()
    out["step_every_outcome_dense"] = run(ticks, b_step, "step", seed=1, dense=True)
    out["step_every_outcome_dense"]["updates"] = int(b_step.learner.updates)
    b_live = fresh()
    out["live_arousal_dense_2"] = run(ticks, b_live, "live", seed=1, dense=True)
    out["live_arousal_dense_2"]["updates"] = int(b_live.learner.updates)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
