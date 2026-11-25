#!/usr/bin/env bash
# Usage: KAGGLE_ENV=<file with KAGGLE_API_TOKEN=...> bash kaggle/push.sh <run_all.sh stage> [new|version|none]
# smoke runs on CPU as kernel <user>/jefkv-smoke (no GPU queue): push it with gate to catch env errors early.
# Uploads the code as a private dataset <user>/jefkv-code (new = first time, version = update)
# and runs kaggle/run_kaggle.py as private kernel <user>/jefkv-run on a T4 with internet.
# Results: kaggle kernels output <user>/jefkv-run -p <dir>
set -euo pipefail
STAGE=${1:-gate}; MODE=${2:-version}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
K=${KAGGLE:-$(command -v kaggle || echo "$ROOT/.venv/Scripts/kaggle")}
set -a; . "${KAGGLE_ENV:?set KAGGLE_ENV}"; set +a
U=$($K config view | sed -n 's/.*username: //p')
KID=jefkv-run; GPU=true; ACC=(--accelerator NvidiaTeslaT4)
[ "$STAGE" = smoke ] && { KID=jefkv-smoke; GPU=false; ACC=(); }
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
mkdir -p "$S/code/data" "$S/kernel"
cp -r "$ROOT"/{jefkv,scripts,tests,run_all.sh,requirements.txt,README.md,LICENSE} "$S/code/"
cp "$ROOT/data/generate_benchmark.py" "$S/code/data/"
find "$S/code" -name __pycache__ -prune -exec rm -rf {} +
echo "{\"title\":\"jefkv-code\",\"id\":\"$U/jefkv-code\",\"licenses\":[{\"name\":\"MIT\"}]}" > "$S/code/dataset-metadata.json"
sed "s/\"JEFKV_STAGE\", \"gate\"/\"JEFKV_STAGE\", \"$STAGE\"/" "$ROOT/kaggle/run_kaggle.py" > "$S/kernel/run_kaggle.py"
cat > "$S/kernel/kernel-metadata.json" <<EOF
{"id":"$U/$KID","title":"$KID","code_file":"run_kaggle.py","language":"python","kernel_type":"script",
 "is_private":true,"enable_gpu":$GPU,"enable_internet":true,"dataset_sources":["$U/jefkv-code"],
 "competition_sources":[],"kernel_sources":[]}
EOF
# kaggle CLI on Windows breaks on absolute -p paths (upload cache path bug) => cd and use -p .
cd "$S/code"
case "$MODE" in new) $K datasets create -p . --dir-mode zip;; version) $K datasets version -p . --dir-mode zip -m "$STAGE";; esac
until [ "$($K datasets status "$U/jefkv-code" | tail -1)" = ready ]; do sleep 10; done
cd "$S/kernel" && $K kernels push -p . "${ACC[@]}"
