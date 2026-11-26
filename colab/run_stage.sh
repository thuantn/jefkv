#!/usr/bin/env bash
# Run one run_all.sh stage on a free Colab T4 with the Colab CLI (Linux/macOS/WSL only), then fetch the results.
#   uv tool install google-colab-cli      # once; first `colab` call asks for a Google login
#   bash colab/run_stage.sh smoke|gate|seeds|full [session-name]
# Env: EXTRA_TAR="data/factkg" (more repo paths in the code tarball, e.g. data that is not in git),
#      KEEP_ADAPTERS=1 (also download adapter.pt: needed to re-evaluate later).
# Steps: new T4 session -> upload a code tarball -> pip install -> nohup the stage -> poll every 2 min
# (polling also keeps the runtime busy) -> download runs/ (adapters excluded) to runs_colab/<stage>/ -> stop.
# On WSL without a keyring service, install with `--with keyrings.alt` and
# export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring (token stored in a file in $HOME).
set -uo pipefail
STAGE=${1:?stage}; S=${2:-jefkv}
cd "$(dirname "$0")/.."
OUT=runs_colab/$STAGE; mkdir -p "$OUT"
TMP=$(mktemp -d); TGZ=$TMP/code.tgz
trap 'rm -rf "$TMP"' EXIT
tar --exclude=__pycache__ -czf "$TGZ" jefkv scripts tests run_all.sh requirements.txt data/generate_benchmark.py ${EXTRA_TAR:-}
# One %%bash cell in /content/jefkv. A failing cell prints a traceback that repeats the cell source,
# so status markers are built from two strings ("ST""ATE=") and never match the source text itself.
cell() { printf '%%%%bash\ncd /content/jefkv 2>/dev/null || true\n%s\n' "$1" | timeout 600 colab exec -s "$S" 2>&1; }

colab new -s "$S" --gpu T4 < /dev/null 2>&1 | tail -1 || { echo "no T4 session (free-tier GPU quota or capacity); try later"; exit 1; }
colab upload -s "$S" "$TGZ" /content/jefkv_code.tgz 2>&1 | tail -1
cell 'mkdir -p /content/jefkv && tar -xzf /content/jefkv_code.tgz -C /content/jefkv && cd /content/jefkv && pip install -q -r requirements.txt 2>&1 | tail -1; nvidia-smi --query-gpu=name --format=csv,noheader'
cell "export PYTHONPATH=.; nohup bash run_all.sh $STAGE > /content/stage.log 2>&1 &"

# Free runtimes can be reclaimed at any time (seen 25/11/2026: 4 finished runs lost), so fetch the results
# after every finished run, not only at the end.
EXA="--exclude=adapter.pt"; [ "${KEEP_ADAPTERS:-}" = 1 ] && EXA=
fetch() {
  cell "tar -czf /content/runs_out.tgz $EXA --exclude='ckpt_*.pt' runs* outputs* /content/stage.log 2>/dev/null; ls -la /content/runs_out.tgz" >/dev/null
  colab download -s "$S" /content/runs_out.tgz "$OUT/runs_out.tgz" 2>&1 | tail -1 && tar -xzf "$OUT/runs_out.tgz" -C "$OUT"
}
fails=0; got=0
while true; do
  sleep 120
  st=$(cell 'pgrep -f run_all.sh >/dev/null && echo "ST""ATE=RUN" || echo "ST""ATE=END"; echo done_runs=$(grep -ac "wrote runs/" /content/stage.log || true)')
  echo "$(date +%H:%M) $(echo "$st" | grep -oE 'STATE=[A-Z]+|done_runs=[0-9]+' | tr '\n' ' ')"
  if echo "$st" | grep -q "STATE=END"; then break; fi
  n=$(echo "$st" | grep -oE 'done_runs=[0-9]+' | cut -d= -f2)
  if [ -n "$n" ] && [ "$n" -gt "$got" ]; then fetch && got=$n; fi
  if echo "$st" | grep -q "STATE=RUN"; then fails=0; else
    fails=$((fails+1)); echo "poll failed ($fails/3)"
    # Gives up after 3 failed polls; finished runs are already in $OUT. The session is kept in case it is alive.
    [ $fails -ge 3 ] && { echo "giving up polling; results so far in $OUT; session '$S' kept (colab stop -s $S)"; exit 1; }
  fi
done

cell 'grep -aE "Traceback|Error" /content/stage.log | tail -5'
# Stop (and lose the VM's files) only once the results are safely on this machine.
if fetch; then
  ls "$OUT"
  colab stop -s "$S" < /dev/null 2>&1 | tail -1
else
  echo "download failed; session '$S' kept: retry the download, then colab stop -s $S"; exit 1
fi
