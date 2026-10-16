#!/usr/bin/env bash
# G7: snapshot, re-run the pipeline, diff. Also: is the PDF byte-reproducible?
set -uo pipefail
cd "$(dirname "$0")"
SNAP=/Users/sheng/.hermes/cache/scratch/ep4nn-snap
rm -rf "$SNAP"; mkdir -p "$SNAP"
cp -R docs/assets "$SNAP/assets"; cp -R docs/receipts "$SNAP/receipts"

bash run_audit_fixes.sh >"$SNAP/rerun.log" 2>&1
env PYTHONPATH=/Users/sheng/math/cadence/src \
  /Users/sheng/.hermes/cache/scratch/cad-venv/bin/python probes/probe_readout_response.py >>"$SNAP/rerun.log" 2>&1

echo "== diff of the artefacts (assets + receipts)"
diff -rq "$SNAP/assets" docs/assets && echo "assets: identical"
diff -rq "$SNAP/receipts" docs/receipts && echo "receipts: identical"

echo "== is the PDF byte-reproducible under SOURCE_DATE_EPOCH?"
export SOURCE_DATE_EPOCH=1760000000
bash docs/paper/build.sh >/dev/null && A=$(shasum -a 256 docs/ep4nn.pdf | cut -d' ' -f1)
bash docs/paper/build.sh >/dev/null && B=$(shasum -a 256 docs/ep4nn.pdf | cut -d' ' -f1)
echo "build1 $A"; echo "build2 $B"; [ "$A" = "$B" ] && echo "pdf: byte-identical" || echo "pdf: differs between builds"

echo "== receipt of the new growth runs"
for f in docs/receipts/grow-demo-*.json; do
  echo "-- $f"; grep -E '"pair_motion"|"readout"|"growths_kept"|"growths_failed"|"ticks"' "$f"
done
