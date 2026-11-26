# FactKG: pre-registered plan and decision rules (written 26/11/2026, before running S0)

Why this file exists: an external review of this repo found confounds that could explain every "JefKV beats RAG"
number so far (adapter depth, the hybrid selector reading the RAG context, train-time shortcut, retriever mismatch,
duplicate gold rows, a biased oracle). Two rounds of brainstorm + cross-review (Gemini, DeepSeek) turned that into
the plan below. The decision rules are fixed here, before the numbers exist; any later change is listed at the end
with its date and reason. A failure is published like a success.

## Known limits of this protocol
- FactKG has no validation split in our converter: S0 is decided on the test split (FactKG dev). S1 carves a
  validation split; every S2 choice (learning rates, variants) is made on it, and test is read once at the end.
- S0 uses the adapters trained on the old data files (79/2,327 test claims have a duplicated gold row; the eval now
  edits every duplicate in the counterfactual).
- 364/2,327 test claims have their gold triple in the train KG; 1,006/2,327 share the head entity. All endpoints are
  also reported on the unseen-head subset.

## S0: eval-only diagnostics on the existing adapters (no training)
Adapters: kv_jef s0-s2, hybrid s0-s2, rag s0-s2 (all with `--rollback_on_collapse 3`), kv_jef randomkv s0.
New eval variants (`scripts/eval.py --diag`):
- `*_qclaim`: the layer-3 selector scores triples with the claim tokens only (not the RAG context, not the fixed
  question). Layer-3 Q/K are frozen, so only the routing changes. It also changes the temperature features, so
  confidence/ECE of these variants are not comparable; accuracy and cf are.
- `hybrid_qclaim_cfkv` / `hybrid_qclaim_cfrag`: the counterfactual edit reaches only the KV path / only the text path.
- `kv_oracle`, `hybrid_oracle`: gold triples forced into the KV top-16 with the prior of the natural top-1 (the old
  oracle gave them a negative prior: a lower bound). `kv_oracle_goldonly`: the KV holds only the gold triples.
- Per-claim predictions in `preds_<variant>.jsonl` (subsets and paired bootstrap without re-running).

Populations: ALL (2,327 claims), RAG-MISS (claims where the trained RAG's dense top-3 does not contain the deciding
gold triple; only there can a KV path add knowledge), UNSEEN-HEAD (head entity absent from the train KG).

### Decision rules (means over 3 seeds; each seed reported)
- **H (hybrid adds knowledge beyond RAG):** GO to the S2 hybrid controls iff on RAG-MISS
  (i) `hybrid_qclaim` acc >= `rag_dense_k3` acc + 5 points, and (ii) `hybrid_qclaim_cfkv` pair accuracy >= 25%.
  If (i) holds only with the full mask (`hybrid_dense_k3`) and not with `qclaim`, the hybrid gain is reported as
  "the selector reads the RAG text", not as KV knowledge.
- **K (where pure KV loses):**
  - `kv_oracle` acc - `kv_full` acc >= 10 points ⇒ the selector is the bottleneck (S2: dense pre-filter / trained selector).
  - `kv_oracle_goldonly` acc - `kv_oracle` acc >= 5 points ⇒ distractors are a bottleneck (S2: hard negatives).
  - `kv_oracle_goldonly` acc < 85 ⇒ the reader is the bottleneck: pure-KV training work stops.
- **Stop:** if H fails and K says "reader", no S2 training: the write-up is the negative result plus the diagnostics.

## S1: data regeneration
Text-key dedup (done in `scripts/factkg_convert.py`), lower-case entity checks in `--grow`, a validation split
disjoint from train and test.

## S2: training controls on the clean data (only the branches S0 sends to), 3 seeds, all runs reported
- hybrid trained with the claim-only selector mask, plain and with tail_shuffle (every triple's tail replaced by
  another tail of the same relation before encoding: same capacity and token layout, no facts). Win claim only if
  hybrid beats both tail_shuffle and the tuned RAG on RAG-MISS by >= 5 acc points and on cf, every seed positive.
- RAG tuned on the validation split (enc_lr x unfrozen layers); strong-retriever RAG (dense + BM25) as an eval variant.
- pure KV with hard negatives and/or the dense pre-filter, only if K points there.
Rollback restarts and collapses are counted and reported per run; no seed is dropped.

## Changes after registration
(none yet)
