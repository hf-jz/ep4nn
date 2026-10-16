#!/usr/bin/env bash
# The unit test, all arms, with the logs kept next to the other receipts.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sheng/math/cadence/src
P=/Users/sheng/.hermes/cache/scratch/cad-venv/bin/python
for arm in "3 1" "6 1" "10 1" "20 1" "10 -1" "20 -1"; do
  set -- $arm
  $P ep4nn/grow.py tone "$1" state 600 "$2" >"docs/receipts/tone-scale$1-sign$2.log" 2>&1
  echo "scale=$1 sign=$2  $(grep -E '^ticks=' "docs/receipts/tone-scale$1-sign$2.log")"
done
for which in default sensory interneuron; do
  $P ep4nn/grow.py "$which" 3 state >/dev/null 2>&1
  cp docs/receipts/grow-demo.json "docs/receipts/grow-demo-$which.json"
done
