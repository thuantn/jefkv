#!/usr/bin/env bash
# Run one run_all.sh stage on a free Colab T4 with the Colab CLI (Linux/macOS/WSL only), then fetch the results.
#   uv tool install google-colab-cli      # once; first `colab` call asks for a Google login
#   bash colab/run_stage.sh smoke|gate|seeds|full [session-name]
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
tar --exclude=__pycache__ -czf "$TGZ" jefkv scripts tests run_all.sh requirements.txt data/generate_benchmark.py
# One %%bash cell in /content/jefkv. A failing cell prints a traceback that repeats the cell source,
# so status markers are built from two strings ("ST""ATE=") and never match the source text itself.
cell() { printf '%%%%bash\ncd /content/jefkv 2>/dev/null || true\n%s\n' "$1" | timeout 600 colab exec -s "$S" 2>&1; }

colab new -s "$S" --gpu T4 < /dev/null 2>&1 | tail -1
colab upload -s "$S" "$TGZ" /content/jefkv_code.tgz 2>&1 | tail -1
cell 'mkdir -p /content/jefkv && tar -xzf /content/jefkv_code.tgz -C /content/jefkv && cd /content/jefkv && pip install -q -r requirements.txt 2>&1 | tail -1; nvidia-smi --query-gpu=name --format=csv,noheader'
cell "export PYTHONPATH=.; nohup bash run_all.sh $STAGE > /content/stage.log 2>&1 &"

fails=0
while true; do
  sleep 120
  st=$(cell 'pgrep -f run_all.sh >/dev/null && echo "ST""ATE=RUN" || echo "ST""ATE=END"; echo done_runs=$(grep -ac "wrote runs/" /content/stage.log || true)')
  echo "$(date +%H:%M) $(echo "$st" | grep -oE 'STATE=[A-Z]+|done_runs=[0-9]+' | tr '\n' ' ')"
  if echo "$st" | grep -q "STATE=END"; then break; fi
  if echo "$st" | grep -q "STATE=RUN"; then fails=0; else
    fails=$((fails+1)); echo "poll failed ($fails/3)"
    # ponytail: gives up after 3 failed polls but keeps the session, so nothing is lost; check it by hand.
    [ $fails -ge 3 ] && { echo "giving up polling; session '$S' kept (colab stop -s $S when done)"; exit 1; }
  fi
done

cell 'grep -aE "Traceback|Error" /content/stage.log | tail -5; tar -czf /content/runs_out.tgz --exclude=adapter.pt runs* outputs* /content/stage.log 2>/dev/null; ls -la /content/runs_out.tgz'
# Stop (and lose the VM's files) only once the results are safely on this machine.
if colab download -s "$S" /content/runs_out.tgz "$OUT/runs_out.tgz" 2>&1 | tail -1 && tar -xzf "$OUT/runs_out.tgz" -C "$OUT"; then
  ls "$OUT"
  colab stop -s "$S" < /dev/null 2>&1 | tail -1
else
  echo "download failed; session '$S' kept: retry the download, then colab stop -s $S"; exit 1
fi
