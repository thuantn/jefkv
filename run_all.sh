#!/usr/bin/env bash
# Usage: bash run_all.sh smoke|gate|warm|seeds|restart|probe|rescue|pairs|rollback|confirm_lr|seeds_lr|fresh|screen_lr|screen_fkv|unf|rag|full
#   warm   re-run the seeds that collapsed in full v1, with --warmup 200 (+ one no-warmup control)
#   seeds  seeds 3-5 of kv_jef / randomkv / hybrid (stability check, run on Colab)
#   restart the 2 runs that collapsed with warmup, re-trained with --restart_on_collapse 2
#   probe  instrumented pair, same recipe: hybrid s1 (collapses) vs s3 (does not), probe() every 25 steps, ckpt at 1000
#   rescue hybrid s1 to step 400 (just before its collapse), then 3 resumes to step 1000: unchanged | LR 3e-4 | new batch order
#   pairs  paired rescue on every known collapse: locate the onset, then 7 one-factor resumes from just before it
#   rollback  the 5 known collapses, full runs with --rollback_on_collapse 3 (detect, go back >=150 steps, nudge 1e-4)
#   confirm_lr  LR 3e-4 on hybrid s0, s2 and randomkv s0, s1 (completes a 3x3 matrix with screen_lr)
#   seeds_lr  seeds 3-5 of kv_jef / randomkv / hybrid with LR 3e-4 (6 seeds per config with screen_lr + confirm_lr)
#   screen_lr / screen_fkv  collapse fixes, one factor each: adapter lr 3e-4 | --freeze_kg_v (kv s0-2, randomkv s2, hybrid s1)
#   fresh  every trained adapter on a fresh test set (seed 7, no shared names) + no-warmup RAG runs
#   smoke  tiny CPU check that every code path runs (minutes)
#   gate   step 1: clean KV vs RandomKV, seed 0. Continue only if the clean loss separates.
#   unf    KV vs RandomKV with the top 2 encoder layers trainable (as RAG trains them), seed 0
#   rag    gate + trained RAG baseline, seed 0 (is the gap to RAG a Jef limit or a KV one?)
#   full   every config x 3 seeds + zero-shot Jef baselines, then results table + plots
set -euo pipefail
STAGE=${1:-gate}
case $STAGE in smoke|gate|warm|seeds|restart|probe|rescue|pairs|rollback|confirm_lr|seeds_lr|fresh|screen_lr|screen_fkv|unf|rag|full) ;; *) echo "unknown stage: $STAGE" >&2; exit 2;; esac
export PYTHONPATH=.
PY=${PY:-python}

if [ "$STAGE" = smoke ]; then
  $PY data/generate_benchmark.py --quick --out_dir data/smoke
  T="--data data/smoke/train.json --steps 4 --batch_size 2"
  E="--test data/smoke/test.json --limit 3"
  $PY scripts/train.py $T --mode kv --save_dir runs_smoke/kv_jef_s0
  $PY scripts/train.py $T --mode kv --random_kv --save_dir runs_smoke/kv_jef_randomkv_s0
  $PY scripts/train.py $T --mode kv --kv_encoder minilm --save_dir runs_smoke/kv_minilm_s0
  $PY scripts/train.py $T --mode rag --save_dir runs_smoke/rag_s0
  $PY scripts/train.py $T --mode hybrid --save_dir runs_smoke/hybrid_s0
  for r in runs_smoke/*_s0; do $PY scripts/eval.py --run "$r" $E; done
  $PY scripts/eval.py --zero_shot base $E --out_dir runs_smoke/zs_base
  $PY scripts/eval.py --zero_shot rag $E --out_dir runs_smoke/zs_rag
  $PY scripts/plot_results.py --runs runs_smoke --out_dir outputs_smoke
  exit 0
fi

[ -f data/processed/test.json ] || $PY data/generate_benchmark.py

train_eval () {  # name, train args...
  local name=$1; shift
  [ -f "runs/$name/eval.json" ] && { echo "skip $name (done)"; return; }
  [ -f "runs/$name/adapter.pt" ] || $PY scripts/train.py "$@" --save_dir "runs/$name"
  $PY scripts/eval.py --run "runs/$name"
}

if [ "$STAGE" = warm ]; then  # seeds that collapsed to 50/50 in full v1 (batch order = seed), with LR warmup
  train_eval kv_jef_nowarm_s2 --mode kv --seed 2 --warmup 0  # no warmup: does the collapse reproduce?
  train_eval kv_jef_w200_s2 --mode kv --seed 2 --warmup 200
  train_eval kv_jef_randomkv_w200_s1 --mode kv --random_kv --seed 1 --warmup 200
  train_eval hybrid_w200_s0 --mode hybrid --seed 0 --warmup 200
  train_eval kv_jef_w200_s0 --mode kv --seed 0 --warmup 200  # a good seed must stay good
  exit 0
fi

if [ "$STAGE" = screen_lr ] || [ "$STAGE" = screen_fkv ]; then  # single-factor collapse fixes
  if [ "$STAGE" = screen_lr ]; then tag=lr3e4; flag="--lr 3e-4"; else tag=fkv; flag="--freeze_kg_v"; fi
  for s in 0 1 2; do train_eval kv_jef_${tag}_s$s --mode kv --seed $s $flag; done
  train_eval kv_jef_randomkv_${tag}_s2 --mode kv --random_kv --seed 2 $flag  # collapsed with warmup, old recipe
  train_eval hybrid_${tag}_s1 --mode hybrid --seed 1 $flag                    # collapsed with warmup, old recipe
  exit 0
fi

if [ "$STAGE" = fresh ]; then  # final adapters on a fresh test set (new entities) that no design decision saw
  for s in 0 1 2; do train_eval rag_nowarm_s$s --mode rag --seed $s --warmup 0; done
  [ -f data/fresh/test.json ] || $PY data/generate_benchmark.py --seed 7 --out_dir data/fresh       --exclude data/processed/train.json data/processed/test.json
  for r in runs/*_s[0-9]; do
    n=$(basename "$r"); [ -f "$r/adapter.pt" ] || continue
    [ -f "runs/fresh/$n/eval.json" ] || $PY scripts/eval.py --run "$r" --test data/fresh/test.json --out_dir "runs/fresh/$n"
  done
  [ -f runs/fresh/zs_base/eval.json ] || $PY scripts/eval.py --zero_shot base --test data/fresh/test.json --out_dir runs/fresh/zs_base
  exit 0
fi

if [ "$STAGE" = confirm_lr ]; then  # LR 3e-4 candidate, seeds not in screen_lr
  for s in 0 2; do train_eval hybrid_lr3e4_s$s --mode hybrid --seed $s --lr 3e-4; done
  for s in 0 1; do train_eval kv_jef_randomkv_lr3e4_s$s --mode kv --random_kv --seed $s --lr 3e-4; done
  exit 0
fi

if [ "$STAGE" = probe ]; then  # when and in what order a collapse happens (no restart, so s1 is logged to 3000)
  for s in 1 3; do train_eval hybrid_probe_s$s --mode hybrid --seed $s --probe_every 25 --ckpt_at 1000; done
  exit 0
fi

if [ "$STAGE" = pairs ]; then  # which change avoids each known (deterministic) collapse, with the same batches?
  for c in "h1w:700:--mode hybrid --seed 1" "r2w:1700:--mode kv --random_kv --seed 2" "k2n:1700:--mode kv --seed 2 --warmup 0" \
           "r1n:1700:--mode kv --random_kv --seed 1 --warmup 0" "h0n:1700:--mode hybrid --seed 0 --warmup 0"; do
    n=${c%%:*}; rest=${c#*:}; L=${rest%%:*}; A=${rest#*:}; D=runs/pair_$n
    if [ ! -f $D/locate/done ]; then
      $PY scripts/train.py $A --probe_every 25 --stop_on_collapse 4 --stop_at $L --ckpt_at $(seq 100 100 1600) --save_dir $D/locate && \
        touch $D/locate/done
    fi
    pick=$($PY scripts/collapse_onset.py $D/locate)
    [ -n "$pick" ] || { echo "pair $n: no collapse reproduced"; continue; }
    set -- $pick; ck=$1; stop=$2; echo "pair $n: ckpt $ck, stop $stop"
    for f in $D/locate/ckpt_*.pt; do [ "$f" = $D/locate/ckpt_$ck.pt ] || rm -f "$f"; done
    for v in "ctrl" "lr099 --lr 9.9e-4" "noise --perturb 1e-4" "lr3e4 --lr 3e-4" "lr2e3 --lr 2e-3" "rs1 --reseed 1001" "rs2 --reseed 1002"; do
      set -- $v; a=$1; shift
      grep -qs "\"step\": $stop," $D/$a/train_log.jsonl || \
        $PY scripts/train.py $A --resume $D/locate/ckpt_$ck.pt --stop_at $stop --probe_every 10 --step_log --save_dir $D/$a "$@"
    done
  done
  exit 0
fi

if [ "$STAGE" = rollback ]; then  # does detect + roll back + 1e-4 nudge turn every known collapse into a normal run?
  R="--rollback_on_collapse 3"
  train_eval hybrid_rb_s1 --mode hybrid --seed 1 $R
  train_eval kv_jef_randomkv_rb_s2 --mode kv --random_kv --seed 2 $R
  train_eval kv_jef_rb_nowarm_s2 --mode kv --seed 2 --warmup 0 $R
  train_eval kv_jef_randomkv_rb_nowarm_s1 --mode kv --random_kv --seed 1 --warmup 0 $R
  train_eval hybrid_rb_nowarm_s0 --mode hybrid --seed 0 --warmup 0 $R
  exit 0
fi

if [ "$STAGE" = rescue ]; then  # is the step-400 state already doomed, or do the next batches / the LR decide?
  R=runs/hybrid_rescue_s1; P="--mode hybrid --seed 1 --probe_every 25"
  [ -f $R/ckpt_400.pt ] || $PY scripts/train.py $P --ckpt_at 400 --stop_at 400 --save_dir $R
  for v in "ctrl" "lr3e4 --lr 3e-4" "reseed --reseed 1001"; do
    set -- $v; n=$1; shift
    grep -qs '"step": 1000' runs/hybrid_rescue_$n/train_log.jsonl ||       $PY scripts/train.py $P --resume $R/ckpt_400.pt --stop_at 1000 --save_dir runs/hybrid_rescue_$n "$@"
  done
  exit 0
fi

if [ "$STAGE" = restart ]; then  # the two runs that collapsed with warmup, with restart-on-collapse
  train_eval hybrid_rs_s1 --mode hybrid --seed 1 --restart_on_collapse 2
  train_eval kv_jef_randomkv_rs_s2 --mode kv --random_kv --seed 2 --restart_on_collapse 2
  exit 0
fi

if [ "$STAGE" = seeds ] || [ "$STAGE" = seeds_lr ]; then  # extra seeds of every frozen-Jef config: how often does it collapse?
  if [ "$STAGE" = seeds_lr ]; then t=_lr3e4; f="--lr 3e-4"; else t=; f=; fi
  for s in 3 4 5; do
    train_eval kv_jef${t}_s$s --mode kv --seed $s $f
    train_eval kv_jef_randomkv${t}_s$s --mode kv --random_kv --seed $s $f
    train_eval hybrid${t}_s$s --mode hybrid --seed $s $f
  done
  exit 0
fi

if [ "$STAGE" = unf ]; then
  train_eval kv_jef_unf2_s0 --mode kv --unfreeze_top 2 --seed 0
  train_eval kv_jef_unf2_randomkv_s0 --mode kv --random_kv --unfreeze_top 2 --seed 0
  exit 0
fi

train_eval kv_jef_s0 --mode kv --seed 0
train_eval kv_jef_randomkv_s0 --mode kv --random_kv --seed 0
if [ "$STAGE" = rag ]; then  # + trained RAG, and KV with the top 2 encoder layers trainable (as RAG)
  train_eval rag_s0 --mode rag --seed 0
  train_eval kv_jef_unf2_s0 --mode kv --unfreeze_top 2 --seed 0
  train_eval kv_minilm_s0 --mode kv --kv_encoder minilm --seed 0  # AtlasKV KG2KV: value = tail only
fi

if [ "$STAGE" = full ]; then
  for s in 0 1 2; do
    train_eval kv_jef_s$s --mode kv --seed $s
    train_eval kv_jef_randomkv_s$s --mode kv --random_kv --seed $s
    train_eval kv_jef_unf2_s$s --mode kv --unfreeze_top 2 --seed $s
    if [ $s = 0 ]; then train_eval kv_minilm_s0 --mode kv --kv_encoder minilm --seed 0; fi  # pair acc 0 at s0: one seed is enough
    train_eval rag_s$s --mode rag --seed $s
    train_eval rag_nowarm_s$s --mode rag --seed $s --warmup 0  # RAG never collapsed: warmup can only cost it
    train_eval hybrid_s$s --mode hybrid --seed $s
  done
  [ -f runs/zs_base/eval.json ] || $PY scripts/eval.py --zero_shot base
  [ -f runs/zs_rag/eval.json ] || $PY scripts/eval.py --zero_shot rag
fi

$PY scripts/plot_results.py --runs runs --out_dir outputs
