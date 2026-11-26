#!/usr/bin/env bash
# Run run_all.sh stage(s) on one rented Vast.ai GPU, fetch the results after every finished job, destroy the instance.
#   pip install vastai                     # CLI; an SSH public key must be registered on the Vast account
#   VAST_ENV=<file with VAST_API_KEY=...> VAST_KEY=<ssh private key> bash vast/run_stage.sh "smoke bench" [RTX_3090]
# Env: JOBS=N (jobs at once on the GPU, see run_all.sh; the offer then needs >= 2 CPU cores per job), ONLY=<regex>
#      (split a stage across machines), JEFKV_AMP (default fp16 = the T4 numerics of every published run; bf16-capable
#      GPUs would otherwise switch to bf16), UPLOAD_RUNS=<local dir of run dirs with adapter.pt> (copied to runs/, e.g.
#      for `fresh`), MAX_DPH (max $/h, default 0.30), MAX_HOURS (default 6), KEEP_ADAPTERS=1 (also fetch adapter.pt),
#      EXTRA_TAR="data/factkg" (more paths from the repo root to put in the code tarball), IMAGE, VASTAI (CLI path).
# Picks the cheapest verified offer outside China (HF / GitHub can be blocked there) with reliability > 0.98 and
# download <= $5/TB; an offer whose image pull fails is destroyed and the next one is tried. Destroys the instance on
# exit, by id and by a per-call label (stopping would keep billing its disk). In the container a watchdog stops the
# job after MAX_HOURS and, if Vast provides an instance-scoped key, destroys the instance, in case this script dies.
set -uo pipefail
STAGES=${1:?stages}; GPU=${2:-RTX_3090}
cd "$(dirname "$0")/.."
: "${VAST_ENV:?file with VAST_API_KEY}"; : "${VAST_KEY:?SSH private key registered on Vast}"
set -a; . "$VAST_ENV"; set +a
PY=${PY:-.venv/Scripts/python}
V() { "${VASTAI:-.venv/Scripts/vastai.exe}" "$@" --api-key "$VAST_API_KEY"; }
export PYTHONIOENCODING=utf-8
MAX_DPH=${MAX_DPH:-0.30}; MAX_HOURS=${MAX_HOURS:-6}; AMP=${JEFKV_AMP:-fp16}; JOBS=${JOBS:-1}
CPUS=$(( JOBS * 2 > 4 ? JOBS * 2 : 4 ))
# Vast's own images are cached on many hosts (fast start, no Docker Hub pull); torch 2.7.1 as in the local venv
# (transformers 5.x fails on torch 2.5.1). It needs a driver for CUDA 12.9.
IMAGE=${IMAGE:-vastai/pytorch:2.7.1-cu128-cuda-12.9-mini-py312-2026-11-20}; CUDA=${CUDA:-12.9}
OUT=runs_vast/${STAGES// /_}${ONLY:+_part}; mkdir -p "$OUT"
TMP=$(mktemp -d); IID=; LABEL=jefkv-$(date +%s)-$$   # unique per call: cleanup finds our instance even if IID was lost
J() { "$PY" -c "import json,sys; d=json.load(sys.stdin); print($1)" 2>/dev/null; }   # tiny JSON field reader
credit() { V show user --raw 2>/dev/null | J "d.get('credit')"; }
C0=$(credit)
destroy() { V destroy instance "$1" -y >/dev/null 2>&1 && echo "destroyed instance $1" || echo "!! destroy $1 FAILED: destroy it by hand (vastai show instances)"; }
cleanup() {
  local ids
  ids=$( { echo "$IID"; V show instances --raw 2>/dev/null | J "'\n'.join(str(i['id']) for i in d if i.get('label') == '$LABEL')"; } | sort -u | grep -v '^$')
  for i in $ids; do destroy "$i"; done
  echo "credit before $C0, after $(credit)"; rm -rf "$TMP"
}
trap cleanup EXIT

V search offers "gpu_name=$GPU num_gpus=1 verified=true reliability>0.98 rentable=true disk_space>=40 cuda_vers>=$CUDA inet_down>=200 cpu_cores_effective>=$CPUS cpu_ram>=16 dph_total<=$MAX_DPH" \
  -o 'dph+' --raw > "$TMP/offers.json" || { echo "offer search failed"; exit 1; }
"$PY" - "$TMP/offers.json" > "$TMP/ids" <<'EOF'
import json, sys
import os, re
d = [o for o in json.load(open(sys.argv[1], encoding="utf-8"))
     if "CN" not in (o.get("geolocation") or "") and (o.get("inet_down_cost") or 0) <= 0.005]
# PREFER_GEO=<regex on the location, e.g. "VN|TH|TW|SG">: those offers first (uploads from Vietnam to Turkey ran at
# ~90 KB/s, to a Vietnamese host at ~6.7 MB/s)
pref = os.environ.get("PREFER_GEO")
if pref:
    d.sort(key=lambda o: (not re.search(pref, o.get("geolocation") or ""), o["dph_total"]))
for o in d[:4]:
    print(o["id"], f"{o['dph_total']:.3f}", (o.get("geolocation") or "?").replace(" ", ""), int(o.get("cpu_cores_effective") or 0))
EOF
[ -s "$TMP/ids" ] || { echo "no offer under \$$MAX_DPH/h for $GPU with >= $CPUS CPUs and CUDA >= $CUDA"; exit 1; }

rent() {  # offer id -> 0 when the instance is running (IID set), else destroys it and returns 1
  local r st msg
  r=$(V create instance "$1" --image "$IMAGE" --disk 40 --ssh --direct --label "$LABEL" --cancel-unavail --raw 2>&1)
  IID=$(echo "$r" | J "d.get('new_contract') or ''")
  [ -n "$IID" ] || { echo "offer $1 failed: ${r:0:150}"; return 1; }
  echo "$(date +%H:%M) instance $IID: $GPU \$$2/h $3, $4 CPUs"
  for i in $(seq 1 60); do   # image pull is not billed, but a host that cannot pull never starts
    V show instance "$IID" --raw > "$TMP/inst.json" 2>/dev/null
    st=$(J "d.get('actual_status')" < "$TMP/inst.json"); msg=$(J "d.get('status_msg') or ''" < "$TMP/inst.json")
    [ "$st" = running ] && return 0
    case "$msg" in *rror*) echo "   host error: ${msg:0:160}"; break;; esac
    sleep 10
  done
  echo "   instance $IID not running (status $st): next offer"; destroy "$IID"; IID=; return 1
}
while read -r oid dph geo cpus; do rent "$oid" "$dph" "$geo" "$cpus" && break; done < "$TMP/ids"
[ -n "$IID" ] || { echo "could not start any offer"; exit 1; }

url=$(V ssh-url "$IID" 2>/dev/null | tail -1)   # ssh://root@HOST:PORT
HP=${url#ssh://root@}; HOST=${HP%:*}; PORT=${HP##*:}
SSHO=(-i "$VAST_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$TMP/kh" -o ConnectTimeout=20 -o ServerAliveInterval=30)
R() { ssh -n -p "$PORT" "${SSHO[@]}" "root@$HOST" "$@"; }
RT() { timeout 40 ssh -n -p "$PORT" "${SSHO[@]}" "root@$HOST" "$@"; }   # the launch of a nohup job never returns: cut it, the job survives   # -n: a backgrounded remote job must not hold the session open
CP() { scp -q -P "$PORT" "${SSHO[@]}" "$@"; }
VENV='[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate >/dev/null 2>&1 < /dev/null;'   # Vast images keep python
# in a venv; its activate script must not hold the ssh session open (seen: the job-start command never returned)
for i in $(seq 1 30); do R true 2>/dev/null && break; sleep 10; done
R true || { echo "ssh to $HOST:$PORT failed"; exit 1; }
echo "$(date +%H:%M) ssh ok ($HOST:$PORT)"

tar --exclude=__pycache__ -czf "$TMP/code.tgz" jefkv scripts tests run_all.sh requirements.txt data/generate_benchmark.py ${EXTRA_TAR:-}
CP "$TMP/code.tgz" "root@$HOST:/root/code.tgz" || { echo "code upload failed"; exit 1; }
if [ -n "${UPLOAD_RUNS:-}" ]; then
  tar -czf "$TMP/runs_in.tgz" -C "$UPLOAD_RUNS" . && CP "$TMP/runs_in.tgz" "root@$HOST:/root/runs_in.tgz" \
    || { echo "upload of $UPLOAD_RUNS failed"; exit 1; }
  echo "$(date +%H:%M) uploaded $(ls "$UPLOAD_RUNS" | wc -l) run dirs from $UPLOAD_RUNS"
fi
R "$VENV mkdir -p /root/jefkv/runs && tar -xzf /root/code.tgz -C /root/jefkv && cd /root/jefkv && \
   { [ ! -f /root/runs_in.tgz ] || tar -xzf /root/runs_in.tgz -C runs; } && pip install -q -r requirements.txt 2>&1 | grep -v 'as the .root. user' | tail -2; \
   nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__, \"cuda ok\", torch.cuda.is_available())'"
RT "cd /root/jefkv && (sleep $((MAX_HOURS * 3600)); pkill -f '^bash run_all[.]sh'; [ -n \"\${CONTAINER_API_KEY:-}\" ] && curl -s -X DELETE https://console.vast.ai/api/v0/instances/\${CONTAINER_ID}/ -H \"Authorization: Bearer \${CONTAINER_API_KEY}\") >/dev/null 2>&1 < /dev/null & \
   $VENV cd /root/jefkv && nohup env PYTHONPATH=. JEFKV_AMP=$AMP JOBS=$JOBS ONLY='${ONLY:-}' bash -c 'for st in $STAGES; do bash run_all.sh \$st || exit 1; done' > /root/stage.log 2>&1 < /dev/null &"
R 'echo container_key_present=$([ -n "${CONTAINER_API_KEY:-}" ] && echo yes || echo no)'
echo "$(date +%H:%M) started: $STAGES (amp $AMP, jobs $JOBS${ONLY:+, ONLY=$ONLY})"

EXA="--exclude=adapter.pt"; [ "${KEEP_ADAPTERS:-}" = 1 ] && EXA=
fetch() {  # one archive name per instance: parallel calls (ONLY splits) share $OUT but never an archive
  R "cd /root/jefkv && tar -czf /root/out.tgz $EXA --exclude='ckpt_*.pt' runs* outputs* /root/stage.log 2>/dev/null; test -s /root/out.tgz" &&
  CP "root@$HOST:/root/out.tgz" "$OUT/out_$IID.tgz" && tar -xzf "$OUT/out_$IID.tgz" -C "$OUT" 2>/dev/null
}
t0=$(date +%s); got=0; fails=0
while true; do
  sleep 120
  # '^bash run_all[.]sh' matches the stage processes only (not this ssh command, not the watchdog's command line)
  st=$(R "pgrep -f '^bash run_all[.]sh' >/dev/null && echo ST''ATE=RUN || echo ST''ATE=END; echo done_runs=\$(grep -ac 'wrote runs/' /root/stage.log || true); nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null)
  echo "$(date +%H:%M) $(echo "$st" | tr '\n' ' ')"
  echo "$st" | grep -q "STATE=END" && break
  if echo "$st" | grep -q "STATE=RUN"; then fails=0; else
    # ssh failed: a local network blip must not destroy a box that is still training; ask the API first
    api=$(V show instance "$IID" --raw 2>/dev/null | J "d.get('actual_status')")
    fails=$((fails + 1)); echo "   ssh poll failed ($fails), API status: ${api:-unknown}"
    if [ "$api" != running ] && [ $fails -ge 5 ] || [ $fails -ge 30 ]; then echo "lost the instance"; fetch; exit 1; fi
  fi
  n=$(echo "$st" | grep -oE 'done_runs=[0-9]+' | cut -d= -f2)
  if [ -n "$n" ] && [ "$n" -gt "$got" ]; then fetch && got=$n; fi
  [ $(( $(date +%s) - t0 )) -gt $((MAX_HOURS * 3600)) ] && { echo "MAX_HOURS reached"; break; }
done
R "grep -aE 'Traceback|Error|FAILED|^bench ' /root/stage.log | tail -12"
fetch && echo "results in $OUT" || echo "!! final fetch failed (earlier results are in $OUT)"
