#!/usr/bin/env bash
# Are the growth receipts deterministic? Same code, two runs, diff. Timing fields are absent
# from the receipt on purpose; anything that differs here is a real nondeterminism.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sheng/math/cadence/src
P=/Users/sheng/.hermes/cache/scratch/cad-venv/bin/python
A=/Users/sheng/.hermes/cache/scratch/rcpt-a; B=/Users/sheng/.hermes/cache/scratch/rcpt-b
rm -rf "$A" "$B"; mkdir -p "$A" "$B"
for round in "$A" "$B"; do
  for which in default sensory interneuron; do
    $P ep4nn/grow.py "$which" >/dev/null 2>&1
    cp docs/receipts/grow-demo.json "$round/grow-demo-$which.json"
  done
done
diff -rq "$A" "$B" && echo "receipts: identical across two runs"
