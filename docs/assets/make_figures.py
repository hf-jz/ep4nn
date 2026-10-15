"""Figures for the ep4nn page. One command, real data, no hand-edited numbers.

fig1 reads docs/assets/prereq_scan.json, written by probes/probe_readout_response.py.
fig2 and fig3 carry the values printed by probes/probe_scaling.py and
probes/probe_certificate.py, transcribed here with their probe named on the figure.

    python docs/assets/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ACCENT, LIGHT, MUTE, INK = "#1d4e89", "#8fb4e3", "#575c65", "#15171b"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.edgecolor": "#dcdcd6", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlesize": 11.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#ececeb", "grid.linewidth": .8,
})

MIN_RESPONSE, CODE_CAP = 0.05, 0.85
COLOR = {None: MUTE, "class_interneuron": LIGHT, "class_command": ACCENT}
LABEL = {None: "no class gain", "class_interneuron": "interneuron gain", "class_command": "command gain"}


def fig1() -> None:
    rows = json.loads((HERE / "prereq_scan.json").read_text())["rows"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": [1, 1]})
    for cls in COLOR:
        pts = [(r["row_sum"], r["spread"]) for r in rows if r["gain_class"] == cls]
        ax.scatter(*zip(*pts), s=58, c=COLOR[cls], label=LABEL[cls],
                   edgecolors="white", linewidths=.8, zorder=3)
    ax.axhline(MIN_RESPONSE, ls="--", lw=1.1, c=INK, zorder=2)
    ax.text(0.11, MIN_RESPONSE * 1.06, "growth gate  min_response = 0.05",
            fontsize=9.5, color=INK)
    best = max(rows, key=lambda r: r["spread"])
    cls = "no class gain" if best["gain_class"] is None else f"{best['gain_class'][6:]} gain"
    ax.annotate(f"best of {len(rows)}: {best['spread']:.4f}\n(row sum {best['row_sum']}, {cls} x{best['class_scale']:g})",
                xy=(best["row_sum"], best["spread"]), xytext=(0.95, 0.030),
                fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTE, lw=.9))
    ax.set(xlabel="row sum of the declared operating point", ylabel="readout spread (mean max - min)",
           title="The answer barely moves", ylim=(-0.003, 0.060))
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    for cls in COLOR:
        pts = [(r["spread"], r["collision_max"]) for r in rows if r["gain_class"] == cls]
        bx.scatter(*zip(*pts), s=58, c=COLOR[cls], edgecolors="white", linewidths=.8, zorder=3)
    bx.axhline(CODE_CAP, ls="--", lw=1.1, c=INK)
    bx.text(0.0045, CODE_CAP - 0.055, "collision gate  code_cap = 0.85", fontsize=9.5, color=INK)
    bx.axhline(1.0, lw=.8, c="#dcdcd6")
    bx.set(xlabel="readout spread", ylabel="largest |cosine| between context answers",
           title="and every answer points the same way", ylim=(0.6, 1.03))
    fig.tight_layout()
    fig.savefig(HERE / "fig1_prereq_scan.png", dpi=100)
    plt.close(fig)


def fig1a() -> None:
    """Hero figure. Same 1280x580 box as the reference page's hero, single panel so the
    type survives the narrow column."""
    rows = json.loads((HERE / "prereq_scan.json").read_text())["rows"]
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    plt.rcParams.update({"font.size": 15})
    for cls in COLOR:
        pts = [(r["row_sum"], r["spread"]) for r in rows if r["gain_class"] == cls]
        ax.scatter(*zip(*pts), s=220, c=COLOR[cls], label=LABEL[cls], edgecolors="white", linewidths=1.4, zorder=3)
    ax.axhline(MIN_RESPONSE, ls="--", lw=2, c=INK, zorder=2)
    ax.text(0.115, MIN_RESPONSE + 0.0035, "growth gate  min_response = 0.05", fontsize=15, color=INK)
    best = max(rows, key=lambda r: r["spread"])
    ax.annotate(f"best of {len(rows)}: {best['spread']:.4f}", xy=(best["row_sum"], best["spread"]),
                xytext=(0.62, 0.060), fontsize=15, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=1.2))
    ax.text(0.115, 0.068, "collision cosine 1.000 at every point", fontsize=15, color=MUTE)
    ax.set(xlabel="row sum of the declared operating point", ylabel="readout spread",
           ylim=(-0.004, 0.072), xlim=(0, 2.15))
    ax.legend(frameon=False, fontsize=15, loc="upper right")
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "fig1a_hero.png", dpi=100)
    plt.close(fig)


def fig2() -> None:
    # probes/probe_scaling.py, row-normalised 0.5, single thread: neurons, synapses, us/sweep, settle ms
    data = [(302, 3_137, 27.2, 4.89, "blocked dense"), (1_000, 10_337, 161.4, 5.29, "blocked dense"),
            (10_000, 103_943, 193.0, 6.48, "CSR"), (100_000, 1_239_921, 2_185.8, 77.4, "CSR")]
    x = [d[0] for d in data]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.plot(x, [d[2] for d in data], "o-", c=ACCENT, lw=2, ms=7, label="per sweep, measured")
    ax.plot(x, [d[3] * 1000 / 32 for d in data], "s--", c=LIGHT, lw=1.8, ms=6,
            label="per sweep implied by the settle")
    for n, s, us, ms, tr in data:
        ax.annotate(f"{us:,.1f} us\n{tr}\nsettle {ms} ms", (n, us), textcoords="offset points",
                    xytext=(-12, 10) if n >= 100_000 else (8, 8),
                    ha="right" if n >= 100_000 else "left", fontsize=9, color=MUTE)
    ax.set(xscale="log", yscale="log", xlabel="neurons", ylabel="microseconds per sweep",
           title="A sparse graph settles 100,000 neurons in 77 ms\nsource: probes/probe_scaling.py",
           ylim=(1, 8000), xlim=(250, 200_000))
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.axhline(1000 / 20 / 32, ls=":", lw=1.2, c=INK)
    ax.text(270, 1000 / 20 / 32 * 1.35, "20 Hz control budget at 32 sweeps", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_scaling.png", dpi=100)
    plt.close(fig)


def fig3() -> None:
    # probes/probe_certificate.py on the worm graph at row sum 0.5
    names = ["global\nbound q", "row bound\n(fails)", "rho(K)\ndefault",
             "rho(J)\ndeclared", "witness\nalpha", "needed\n(1)"]
    vals = [88.0, 1.186, 0.529, 0.931, 0.696, 1.0]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    bars = ax.bar(names, vals, color=[MUTE, "#c9b7a7", LIGHT, ACCENT, "#4a7fb5", "#dcdcd6"], width=.62)
    ax.set(yscale="log", ylabel="contraction factor (log scale)",
           title="A certificate exists where the global bound fails\nsources: probe_certificate.py, probe_audit_claims.py",
           ylim=(0.4, 300))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:g}", ha="center", fontsize=10.5, color=INK)
    ax.annotate("292 of 302 rows at or above 1:\nno certificate", xy=(0, 88), xytext=(0.42, 170),
                fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTE, lw=.9))
    ax.annotate("certificate: spectral radius of the\nmajorant at both operating points,\nbounded by the weighted witness",
                xy=(2, 0.529), xytext=(1.55, 2.6), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=.9))
    fig.tight_layout()
    fig.savefig(HERE / "fig3_certificate.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    fig1a(), fig1(), fig2(), fig3()
    for p in sorted(HERE.glob("fig*.png")):
        print(p.name, p.stat().st_size)
