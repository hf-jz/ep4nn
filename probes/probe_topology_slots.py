"""P4 (cheap half): are memory slots topological holes?

Bianconi's higher-order result says dynamics can live on edges and triangles, and that
synchronisation of a k-dimensional signal needs a k-dimensional hole. The tempting shortcut for
ep4nn is "if the basin count is the Betti number, slots are countable without simulating".

That shortcut has a one-line refutation if it is wrong: the cycle rank of a graph is
b1 = E - V + b0, which grows with the edge count (density x N^2), while the measured slot count is
flat in N (§18, §20). This probe computes the invariants of exactly the structure families that
produced the slot measurements, so the comparison is on the same graphs, not on a proxy.

What this does NOT do: build a Hodge Laplacian or run higher-order dynamics. That would be a new
substrate (state on edges and triangles), not a measurement on this one, and it belongs to a
separate project until something here says it pays.

    python probes/probe_topology_slots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_basins import genome  # noqa: E402

from cadence.genome import develop  # noqa: E402

SIZES = (32, 64, 128, 256)
INPUT_DIV = 4


def invariants(conn) -> dict:
    n = conn.n
    edges = {(min(int(a), int(b)), max(int(a), int(b)))
             for a, b in zip(conn.pre, conn.post) if int(a) != int(b)}
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    seen = np.zeros(n, bool)
    b0 = 0
    for start in range(n):
        if seen[start]:
            continue
        b0 += 1
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

    v_alive = int(sum(1 for i in range(n) if adj[i]))
    b1 = len(edges) - v_alive + b0
    triangles = sum(len(adj[u] & adj[v]) for u, v in edges) // 3
    return {"n": n, "vertices_with_edges": v_alive, "edges": len(edges),
            "b0": b0, "b1": b1, "triangles": triangles,
            "b1_per_vertex": round(b1 / max(v_alive, 1), 2)}


def held_slots(path: Path) -> dict:
    """Largest cue count where every smaller count also passed (monotone reading of §20)."""
    data = json.loads(path.read_text())
    out = {}
    for row in data["rows"]:
        key = (row["hidden"], row["n_neurons"])
        out.setdefault(key, []).append((row["cues"], bool(row["separable"])))
    held = {}
    for key, rows in out.items():
        best = 0
        for cues, ok in sorted(rows):
            if not ok:
                break
            best = cues
        held[key] = best
    return held


def main() -> None:
    slots = held_slots(HERE.parent / "docs" / "assets" / "basins_scaled_seed0.json")
    result = {"probe": "probe_topology_slots.py", "input_div": INPUT_DIV, "rows": []}
    print(f"{'hidden':>6} {'neurons':>8} {'edges':>7} {'b0':>3} {'b1':>7} {'tri':>7} "
          f"{'b1/V':>6} {'slots':>6}")
    for hidden in SIZES:
        conn = develop(genome(hidden, inputs=max(14, hidden // INPUT_DIV), seed=0), seed=0)
        inv = invariants(conn)
        slot = slots.get((hidden, inv["n"]), None)
        inv["slots_measured"] = slot
        result["rows"].append(inv)
        print(f"{hidden:>6} {inv['n']:>8} {inv['edges']:>7} {inv['b0']:>3} {inv['b1']:>7} "
              f"{inv['triangles']:>7} {inv['b1_per_vertex']:>6} {str(slot):>6}")

    first, last = result["rows"][0], result["rows"][-1]
    b1_growth = last["b1"] / max(first["b1"], 1)
    slot_growth = (last["slots_measured"] or 0) / max(first["slots_measured"] or 1, 1)
    result["b1_growth"] = round(b1_growth, 1)
    result["slot_growth"] = round(slot_growth, 1)
    result["verdict"] = (
        "REFUTED: b1 与槽位不同构——b1 随边数增长 %.1fx，槽位只有 %.1fx，且槽位已到测量上限"
        % (b1_growth, slot_growth)
        if b1_growth > 2 * slot_growth else
        "POSSIBLE: b1 与槽位同阶增长，值得建 Hodge 动力学继续")

    assert all(row["b1"] >= 0 for row in result["rows"])
    assert result["rows"][-1]["n"] > result["rows"][0]["n"]

    out = HERE.parent / "docs" / "assets" / "topology_slots.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"\nb1 growth {b1_growth:.1f}x vs slot growth {slot_growth:.1f}x")
    print(result["verdict"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
