#!/usr/bin/env bash
# Does the generator, once it reads the distinction where it actually lives, ever keep a growth?
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sheng/math/cadence/src
P=/Users/sheng/.hermes/cache/scratch/cad-venv/bin/python
S=/Users/sheng/.hermes/cache/scratch
for src in state readout; do
  echo "=== difference_source=$src ==="
  $P ep4nn/grow.py default "$src" >"$S/grow-$src.log" 2>&1
  grep -E "kept|rolled_back" "$S/grow-$src.log" | head -6
  tail -2 "$S/grow-$src.log" | head -1
  cp docs/receipts/grow-demo.json "docs/receipts/grow-demo-default-$src.json"
  echo "   kept=$(grep -c 'kept' "$S/grow-$src.log") rolled_back=$(grep -c 'rolled_back' "$S/grow-$src.log")"
done
