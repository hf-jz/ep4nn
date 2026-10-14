"""Probe 3: transport path costs, and an end-to-end tick with an accelerated settle.

Part A  measures the three transport implementations the library can use
        (dense block matmul / scipy CSR / segmented reduceat) on three graph sizes.
Part B  replaces only the numerical settle (Brain.equilibrate) with an
        Anderson-mixed iteration of the same one-step map, keeps the library's
        own residual check and qualification, and reruns the same life.
"""

from __future__ import annotations

import json
import time

import numpy as np

from cadence import Brain
from cadence.brain import BrainState, Equilibrium

OUT: dict[str, object] = {}


def transport_bench(brain: Brain, reps: int = 500) -> dict:
    core = brain.brain
    n, syn = core.connectome.n, core.connectome.synapses
    s = np.full((1, n), 0.1)
    res = {"neurons": int(n), "synapses": int(syn), "blocked": bool(core._blocked)}
    # dense blocks
    if core._blocks is not None:
        from cadence.blocks import BlockTransport

        bt = BlockTransport(core.layout, core._blocks)
        bt.synaptic_input(s)
        t0 = time.perf_counter()
        for _ in range(reps):
            bt.synaptic_input(s)
        res["blocked_dense_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    # scipy csr as the library builds it
    from cadence.sparse import transport

    csr = transport(core.connectome, core.weights)
    res["csr_available"] = csr is not None
    if csr is not None:
        csr.dot(s.T).T
        t0 = time.perf_counter()
        for _ in range(reps):
            csr.dot(s.T).T
        res["csr_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    # library _synaptic_input with no blocks argument (forces csr/numpy path)
    core._synaptic_input(s)
    t0 = time.perf_counter()
    for _ in range(reps):
        core._synaptic_input(s)
    res["library_synaptic_input_noblocks_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    # raw segmented reduceat
    pre, starts, targets, w = core.connectome.pre, core._starts, core._neurons_with_synaptic_input, core.weights
    t0 = time.perf_counter()
    for _ in range(reps):
        msg = s[:, pre] * w
        out = np.zeros((1, n))
        out[:, targets] = np.add.reduceat(msg, starts, axis=1)
    res["reduceat_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    # one full library sweep
    drive = np.zeros((1, n))
    core.settle_batch(drive, steps=1, state=None)
    t0 = time.perf_counter()
    for _ in range(reps):
        core.settle_batch(drive, steps=1, state=None)
    res["library_sweep_us"] = round(1e6 * (time.perf_counter() - t0) / reps, 1)
    return res


def make_accelerated_equilibrate(m: int = 5, lam: float = 1e-8, fallback=None):
    """A drop-in Brain.equilibrate that mixes the same one-step map.

    If the mixed iteration does not qualify inside the same budget, the original
    library solver finishes the attempt, so the qualification contract is kept.
    """

    def equilibrate(self, drive, *, budget=512, chunk=32, tolerance=1e-5, state=None, mask=None,
                    nudge=None, damping=0):
        d = np.asarray(drive, float)
        if d.ndim == 1:
            d = d[None, :]

        def one(x):
            return self.settle_batch(d, steps=1, state=x, mask=mask, nudge=nudge)

        def res(x):
            return float(np.max(self.residual(d, x, mask=mask, nudge=nudge)))

        cur = self.settle_batch(d, steps=0, state=state, mask=mask, nudge=nudge)
        err = res(cur)
        checks = 1
        used = 0
        if err <= tolerance:
            return Equilibrium(cur, np.array([err]), 0, tolerance, residual_checks=checks)
        v = np.array(cur.v, dtype=float, copy=True, ndmin=2)
        f = one(BrainState(v=v, activation=cur.activation, adaptation=np.array(cur.adaptation), steps=0))
        fv = np.array(f.v, dtype=float, ndmin=2)
        used = 1
        rs = [(fv - v).ravel().copy()]
        fs = [fv.ravel().copy()]
        err = res(f)
        checks += 1
        while used < budget and err > tolerance:
            k = min(m, len(rs) - 1)
            cand = None
            if k >= 1:
                base, fbase = rs[-1], fs[-1]
                dif = np.stack([rs[-1 - j] - base for j in range(1, k + 1)], axis=1)
                fdif = np.stack([fs[-1 - j] - fbase for j in range(1, k + 1)], axis=1)
                try:
                    beta = np.linalg.solve(dif.T @ dif + lam * np.eye(k), -dif.T @ base)
                    cand = (fbase + fdif @ beta).reshape(v.shape)
                except np.linalg.LinAlgError:
                    cand = None
            if cand is None or not np.isfinite(cand).all():
                cand = fv
            ns = BrainState(v=cand, activation=self.neuron_model.activation(cand),
                            adaptation=np.array(cur.adaptation), steps=0)
            fc = np.array(one(ns).v, dtype=float, ndmin=2)
            used += 1
            if np.linalg.norm(fc - cand) <= np.linalg.norm(fv - v):
                v, fv = cand, fc
            else:
                plain = BrainState(v=fv, activation=self.neuron_model.activation(fv),
                                   adaptation=np.array(cur.adaptation), steps=0)
                v, fv = fv, np.array(one(plain).v, dtype=float, ndmin=2)
                used += 1
            rs.append((fv - v).ravel().copy())
            fs.append(fv.ravel().copy())
            if len(rs) > m + 2:
                rs.pop(0)
                fs.pop(0)
            final = BrainState(v=v, activation=self.neuron_model.activation(v),
                               adaptation=np.array(cur.adaptation), steps=used)
            err = res(final)
            checks += 1
        final = BrainState(v=v, activation=self.neuron_model.activation(v),
                           adaptation=np.array(cur.adaptation), steps=used)
        out = Equilibrium(final, np.array([res(final)]), used, tolerance, residual_checks=checks)
        if not bool(np.all(out.qualified)) and fallback is not None:
            return fallback(self, drive, budget=budget, chunk=chunk, tolerance=tolerance,
                            state=state, mask=mask, nudge=nudge, damping=damping)
        return out

    return equilibrate


def live_run(b: Brain, ticks: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    obs = np.eye(len(b.sensory_index))
    sweeps, times, residuals, qualified = [], [], [], 0
    b.live(obs[[0]])
    for _ in range(ticks):
        x = obs[rng.integers(0, len(b.sensory_index))][None, :]
        r = float(rng.random() < 0.4)
        t0 = time.perf_counter()
        b.live(x, reward=np.array([r]))
        times.append(time.perf_counter() - t0)
        s = b.last_settlement
        sweeps.append(int(s["steps"]))
        residuals.append(float(s["max_residual"]))
        qualified += int(bool(s["qualified"]))
    return {
        "ticks": ticks,
        "ms_per_tick_mean": round(1000 * float(np.mean(times)), 3),
        "ms_per_tick_p95": round(1000 * float(np.percentile(times, 95)), 3),
        "ticks_per_second": round(1.0 / float(np.mean(times)), 1),
        "sweeps_mean": round(float(np.mean(sweeps)), 2),
        "sweeps_max": int(np.max(sweeps)),
        "residual_worst": float(np.max(residuals)),
        "qualified_all": bool(qualified == ticks),
    }


def main() -> None:
    for modules in [(64, 32, 16), (256, 128, 64)]:
        b = Brain.compose(inputs=8, actions=4, modules=modules, arousal=True, seed=3)
        OUT[f"transport{modules}"] = transport_bench(b, reps=300)

    # end-to-end: library settle vs the same settle under Anderson mixing
    b = Brain.compose(inputs=8, actions=4, modules=(64, 32, 16), arousal=True, seed=3)
    OUT["live_library"] = live_run(b, 80)

    b2 = Brain.compose(inputs=8, actions=4, modules=(64, 32, 16), arousal=True, seed=3)
    core_cls = type(b2.brain)
    original = core_cls.equilibrate
    core_cls.equilibrate = make_accelerated_equilibrate(fallback=original)
    OUT["live_anderson"] = live_run(b2, 80)
    core_cls.equilibrate = original
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
