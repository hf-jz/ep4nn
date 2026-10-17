#!/usr/bin/env bash
# Verify the deliverable end to end: the two self-checks, the receipt, and the live site.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/Users/sheng/math/cadence/src
P="${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3 )}"   # >= 3.10 with numpy
OUT=/Users/sheng/.hermes/cache/scratch/ep4nn-index.html

echo "== receipt self-check"; $P ep4nn/receipt.py
echo "== grow.py self-check"; $P ep4nn/grow.py | tail -5
echo "== receipt file"; ls -l docs/receipts/ 2>/dev/null; head -24 docs/receipts/grow-demo.json 2>/dev/null

echo "== waiting for the Pages build"
for i in 1 2 3 4 5 6; do
  sleep 25
  code=$(curl -sS -o "$OUT" -w "%{http_code}" https://hf-jz.github.io/ep4nn/)
  echo "attempt $i: status=$code bytes=$(wc -c <"$OUT" | tr -d ' ')"
  [ "$code" = "200" ] && break
done
echo "== what is served"
grep -o "<title>[^<]*</title>" "$OUT" | head -1
grep -c "<h2" "$OUT" || true
curl -sS -o /dev/null -w "paper.md=%{http_code}\n" https://hf-jz.github.io/ep4nn/paper.md
curl -sS -o /dev/null -w "style.css=%{http_code}\n" https://hf-jz.github.io/ep4nn/style.css
