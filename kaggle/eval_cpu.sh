#!/usr/bin/env bash
# Eval-only jobs on Kaggle CPU kernels (no GPU queue), one kernel per adapter, all in parallel.
# Usage: KAGGLE_ENV=<file with KAGGLE_API_TOKEN=...> bash kaggle/eval_cpu.sh <tag> "<eval.py args>" <adapter dir>... [new|version]
#   e.g. bash kaggle/eval_cpu.sh rrf "--test data/factkg/test.json --diag --variants ^rag_rrf_k3$" runs/fk_rag_unf6_s0 runs/fk_rag_unf6_s1 new
# Uploads code + data/factkg + the adapters as a PRIVATE dataset <user>/jefkv-<tag> (FactKG is CC BY-NC-SA: never public),
# then pushes private CPU kernels <user>/jefkv-<tag>-<n>. Fetch: kaggle kernels output <user>/jefkv-<tag>-<n> -p <dir>.
set -euo pipefail
TAG=$1; ARGS=$2; shift 2
MODE=version; case "${!#}" in new|version) MODE=${!#}; set -- "${@:1:$#-1}";; esac
ROOT=$(cd "$(dirname "$0")/.." && pwd)
K=${KAGGLE:-$ROOT/.venv/Scripts/kaggle}
set -a; . "${KAGGLE_ENV:?set KAGGLE_ENV}"; set +a
U=$($K config view | sed -n 's/.*username: //p')
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
mkdir -p "$S/code/data"
cp -r "$ROOT"/{jefkv,scripts,requirements.txt} "$S/code/"
cp -r "$ROOT/data/factkg" "$S/code/data/"
for d in "$@"; do mkdir -p "$S/code/$d"; cp "$ROOT/$d/adapter.pt" "$S/code/$d/"; done
find "$S/code" -name __pycache__ -prune -exec rm -rf {} +
echo "{\"title\":\"jefkv-$TAG\",\"id\":\"$U/jefkv-$TAG\",\"licenses\":[{\"name\":\"other\"}]}" > "$S/code/dataset-metadata.json"
cd "$S/code"  # the CLI breaks on absolute -p paths on Windows
case "$MODE" in new) $K datasets create -p . --dir-mode zip;; version) $K datasets version -p . --dir-mode zip -m "$TAG";; esac
sleep 120  # `datasets status` answers 403 for this private dataset (27/11/2026), so wait a fixed time for processing
n=0
for d in "$@"; do
  n=$((n + 1)); mkdir -p "$S/k$n"
  cat > "$S/k$n/run.py" <<EOF
import glob, os, shutil, subprocess, sys
src = os.path.dirname(glob.glob("/kaggle/input/**/requirements.txt", recursive=True)[0])
shutil.copytree(src, "/kaggle/working/jefkv", dirs_exist_ok=True); os.chdir("/kaggle/working/jefkv")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
env = dict(os.environ, PYTHONPATH=".")
subprocess.run([sys.executable, "scripts/eval.py", "--run", "$d"] + """$ARGS""".split(), check=True, env=env)
for f in glob.glob("runs/**/adapter.pt", recursive=True):
    os.remove(f)
EOF
  echo "{\"id\":\"$U/jefkv-$TAG-$n\",\"title\":\"jefkv-$TAG-$n\",\"code_file\":\"run.py\",\"language\":\"python\",\"kernel_type\":\"script\",\"is_private\":true,\"enable_gpu\":false,\"enable_internet\":true,\"dataset_sources\":[\"$U/jefkv-$TAG\"],\"competition_sources\":[],\"kernel_sources\":[]}" > "$S/k$n/kernel-metadata.json"
  (cd "$S/k$n" && $K kernels push -p .)
  echo "kernel $U/jefkv-$TAG-$n <- $d"
done
