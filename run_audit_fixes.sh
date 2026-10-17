#!/usr/bin/env bash
# Re-run everything the audit's fix list needs, in order, with the outputs kept.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sheng/math/cadence/src
P="${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3 )}"   # >= 3.10 with numpy

echo "== receipt self-check"; $P ep4nn/receipt.py
echo "== audit fixes probe (G1 certified bound, G2 domain, G5 seeds, G4 readout family)"
$P probes/probe_audit_fixes.py 2>&1 | tail -8
for which in default sensory interneuron; do
  echo "== growth demo, readout=$which"
  $P ep4nn/grow.py "$which" 2>&1 | tail -3
  cp docs/receipts/grow-demo.json "docs/receipts/grow-demo-$which.json"
done
