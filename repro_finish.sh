#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sheng/math/cadence/src
P=/Users/sheng/.hermes/cache/scratch/cad-venv/bin/python
B=/Users/sheng/.hermes/cache/scratch/rcpt-b
rm -rf "$B"; mkdir -p "$B"
for which in default sensory interneuron; do
  $P ep4nn/grow.py "$which" >/dev/null 2>&1
  cp docs/receipts/grow-demo.json "$B/grow-demo-$which.json"
done
diff -rq /Users/sheng/.hermes/cache/scratch/rcpt-a "$B" && echo "receipts: IDENTICAL across two runs (deterministic)"
$P docs/assets/make_figures.py
