# Raw results

`<stage>/<run>/eval.json` (all metrics, see `scripts/eval.py`) and `train_log.jsonl` (loss, probes, rollbacks) for the
runs behind the tables in the main README. Adapters are not stored. Stages are the `run_all.sh` stages; the GPU of each
stage: `seeds_lr` Colab T4, `rb_check` / `fresh` / `factkg*` Vast.ai RTX 3090 (fp16, the T4 recipe).

- `fresh/`: adapters trained earlier (T4), evaluated on a fresh synthetic test set (seed 7, no names shared with
  train/test). `fresh/rag_nowarm_s*` were trained for this stage.
- `rb_check/`: the recommended recipe (`--rollback_on_collapse 3`) on seeds 0–5. `kv_jef_randomkv_rbv_s4` is missing:
  its eval ran out of GPU memory three times and the box hit its time limit.
- `seeds_lr/`: LR 3e-4, seeds 3–5.
- `factkg*/`: FactKG (real DBpedia claims). Only metrics: the FactKG data (CC BY-NC-SA 4.0) is never stored here.
  `factkg_s0/`: eval-only S0 of docs/PREREG_factkg.md on the FactKG adapters (kv_jef, hybrid s0-2; RAG s0-2 were
  scored from per-claim predictions, not kept here) and `gates.txt` (every variant x population x seed, decisions).
  `factkg_s0b/`: the post-hoc checks (hybrid with shuffled KV values, RAG with 6 trainable layers, 3 seeds each).
