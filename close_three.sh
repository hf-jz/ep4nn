#!/usr/bin/env bash
# The three items the audit left open, re-measured with the fixed verdict layer.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sheng/math/cadence/src
P="${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3 )}"   # >= 3.10 with numpy
$P probes/probe_closure.py 2>&1 | tail -20
for which in default sensory interneuron; do
  $P ep4nn/grow.py "$which" >/dev/null 2>&1
  cp docs/receipts/grow-demo.json "docs/receipts/grow-demo-$which.json"
done
$P docs/assets/make_figures.py >/dev/null && echo "figures regenerated"
