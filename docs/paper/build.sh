#!/usr/bin/env bash
# Build the paper PDF next to the site index, and fail loudly on an unresolved reference.
# One command: bash docs/paper/build.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$( [ -x ../../.venv/bin/python ] && echo ../../.venv/bin/python || command -v python3 )}"
"$PY" test_paper_numbers.py || { echo "FAIL: a printed number does not match its artefact"; exit 1; }
xelatex -interaction=nonstopmode -halt-on-error ep4nn.tex >/dev/null
xelatex -interaction=nonstopmode -halt-on-error ep4nn.tex >/dev/null

if grep -qE "There were undefined references|Citation .* undefined" ep4nn.log; then
  echo "FAIL: undefined references"; grep -E "Warning|undefined" ep4nn.log | head -10; exit 1
fi
cp ep4nn.pdf ../ep4nn.pdf
echo "pages: $(grep -o "([0-9]* pages" ep4nn.log | tail -1 | tr -d '(')"
if grep -q "Overfull" ep4nn.log; then echo "OVERFULL:"; grep -A2 "Overfull" ep4nn.log | head -12; fi
rm -f ep4nn.aux ep4nn.log ep4nn.out ep4nn.toc ep4nn.pdf
echo "built $(cd .. && pwd)/ep4nn.pdf"
