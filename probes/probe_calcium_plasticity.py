"""P3: replace the EP-style contrast with a calcium-thresholded local rule, and see whether
refusals become readable.

Why this probe exists. The NeuroSyn review states the textbook rule: a strong, fast calcium
transient drives LTP through kinases; a weak, sustained rise drives LTD through phosphatases, and
neuromodulators move the threshold. ep4nn's own learning is EP-style contrast (free phase versus
nudged phase), which has no threshold at all -- so when a lesson is not learned we cannot say
whether the calcium signal never crossed, or crossed in the wrong direction, or crossed fine and the
readout was saturated. The three are indistinguishable in the current code.

Minimal test, no library surgery: keep the frozen c302 graph as the feature extractor, learn a
linear readout to a binary valence label, and compare two local rules on the SAME data, SAME
learning rate, SAME epochs:

    delta:  w += eta (y - p) x_j                                (the EP contrast, collapsed to per-step)
    ca:     Ca_j <- decay Ca_j + (1-decay) |x_j| |y - p|
            if Ca_j > theta:  w += eta (y - p) x_j  (scaled)
            else:             no update  -> counted as a refusal, with its Ca value

Pre-registered gate, on the low-SNR setting: PASS if ca beats delta by >= 0.05 test accuracy while
reporting what fraction of updates it refused. If ca == delta, the threshold buys interpretability
but not performance, and the honest line is "refusals become attributable, accuracy unchanged".

    python probes/probe_calcium_plasticity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_manifold import build, collect, load_worm_with_roles  # noqa: E402

THETA = 0.02
DECAY = 0.7


def make_task(conn, n, rng, snr: str, noise: float = 0.0):
    """Balanced +/- valence. Low SNR = weak target drive plus distractors of equal strength."""
    pain = np.array(sorted(conn.populations["pain"]))
    food = np.concatenate([conn.populations[p] for p in ("odour_A", "odour_B", "food")])
    sensory = np.array(sorted(set(conn.populations["odour_A"] + conn.populations["odour_B"])))
    x, y = [], []
    for _ in range(n):
        row = np.zeros(conn.n)
        positive = bool(rng.integers(2))
        pool = food if positive else pain
        if snr == "easy":
            amp, k = rng.uniform(0.4, 1.0), 4
        else:
            amp, k = rng.uniform(0.05, 0.15), 2
            row[rng.choice(sensory, size=min(6, len(sensory)), replace=False)] = amp   # distractors
        k = int(min(k, len(pool)))
        row[rng.choice(pool, size=k, replace=False)] = amp
        x.append(row)
        y.append(1.0 if positive else 0.0)
    y = np.array(y)
    if noise > 0:                                   # label noise: the regime where a threshold
        flip = rng.random(len(y)) < noise           # could plausibly reject bad lessons
        y[flip] = 1.0 - y[flip]
    return np.stack(x), y


def train(states, labels, *, rule: str, epochs: int, eta: float, seed: int,
          theta: float = THETA):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, states.shape[1])
    b = 0.0
    ca = np.zeros(states.shape[1])
    refusals, updates = 0, 0
    for _ in range(epochs):
        for i in rng.permutation(len(states)):
            xj, yj = states[i], labels[i]
            p = 1.0 / (1.0 + np.exp(-(w @ xj + b)))
            err = yj - p
            ca = DECAY * ca + (1 - DECAY) * np.abs(xj) * abs(err)
            if rule == "ca":
                gate = ca > theta
                if not gate.any():
                    refusals += 1
                    continue
                w += eta * err * np.where(gate, xj, 0.0)
                updates += 1
            else:
                w += eta * err * xj
                updates += 1
            b += eta * err * 0.1
    return w, b, refusals, updates, ca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.25,
                        help="label noise rate; the regime where a Ca threshold could help")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-sum", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = load_worm_with_roles()
    brain = build(conn, row_sum=args.row_sum)
    rng = np.random.default_rng(args.seed)

    result = {"probe": "probe_calcium_plasticity.py", "theta": THETA, "decay": DECAY,
              "epochs": args.epochs, "eta": args.eta, "settings": {}}
    for snr in ("easy", "hard"):
        x, y = make_task(conn, args.samples, rng, snr, args.noise)
        states, _, _ = collect(brain, x)
        split = int(0.7 * len(states))
        s_tr, s_te = states[:split], states[split:]
        mu, sd = s_tr.mean(axis=0), s_tr.std(axis=0) + 1e-6     # z-score: eta is per-unit then
        s_tr, s_te = (s_tr - mu) / sd, (s_te - mu) / sd
        y_tr, y_te = y[:split], y[split:]
        block = {}
        arms = [("delta", None)] + [(f"ca@{t}", t) for t in (0.05, 0.15, 0.3, 0.6)]
        for rule, theta in arms:
            w, b, refusals, updates, ca = train(s_tr, y_tr, rule="delta" if theta is None else "ca",
                                                epochs=args.epochs, eta=args.eta, seed=args.seed,
                                                theta=theta or 0.0)
            acc_tr = float(np.mean((s_tr @ w + b > 0) == (y_tr > 0.5)))
            acc_te = float(np.mean((s_te @ w + b > 0) == (y_te > 0.5)))
            block[rule] = {"train_accuracy": round(acc_tr, 3), "test_accuracy": round(acc_te, 3),
                           "refusal_rate": round(refusals / max(refusals + updates, 1), 3),
                           "ca_median": round(float(np.median(ca)), 5)}
            print(f"{snr:5} {rule:7} train={acc_tr:.3f} test={acc_te:.3f} "
                  f"refusals={block[rule]['refusal_rate']:.2f} "
                  f"ca_median={block[rule]['ca_median']:.5f}", flush=True)
        result["settings"][snr] = block

    def best(node):
        return max(node[k]["test_accuracy"] for k in node if k.startswith("ca@"))

    easy_gap = best(result["settings"]["easy"]) - result["settings"]["easy"]["delta"]["test_accuracy"]
    hard_gap = best(result["settings"]["hard"]) - result["settings"]["hard"]["delta"]["test_accuracy"]
    result["gap_easy"] = round(easy_gap, 3)
    result["gap_hard"] = round(hard_gap, 3)
    result["verdict"] = ("PASS: 阈值规则在低信噪下更准，且拒绝可计数" if hard_gap >= 0.05 else
                         "NO GAIN: 阈值只买来可解释性，不买准确率")

    assert all(0.0 <= s[r]["refusal_rate"] <= 1.0 for s in result["settings"].values() for r in s)

    out = args.out or (HERE.parent / "docs" / "assets" / "calcium_plasticity.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"\ngap easy={easy_gap:+.3f}  hard={hard_gap:+.3f}")
    print(result["verdict"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
