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
  oracle gave them a negative prior: a lower bound); `hybrid_oracle` also puts them first in the retrieved text.
  `kv_oracle_goldonly`: the KV holds only the gold triples.
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
- 26/11/2026, after S0: the Stop rule needed K = "reader", but no K branch fired (on ALL; the rules did not name the
  population for K: on RAG-MISS `kv_oracle_goldonly` is 74.3 < 85, on UNSEEN-HEAD 85.8). We stop anyway (reason in the
  Outcome) and record it here as a deviation.
- 26/11/2026, after S0, before running them: two post-hoc checks of the Outcome's explanation, 3 seeds each, same data
  and recipe (`run_all.sh factkg_s0b`): hybrid trained with shuffled KV values (does the hybrid use the KG facts, or the
  KV branch only as a text-conditioned adapter?) and RAG with 6 unfrozen layers (is the hybrid's lead adapter capacity?).
  Reported as post-hoc, not as a registered test.

## Outcome of S0 (26/11/2026, 3 seeds each; numbers: `results/factkg_s0/gates.txt`)
- **H: FAIL.** On RAG-MISS (290 claims): `hybrid_qclaim` acc 92.3 vs `rag_dense_k3` 89.2 (+3.1, rule: +5);
  `hybrid_qclaim_cfkv` pair accuracy 2.1 (rule: >= 25). Editing only the KV changes the hybrid's answer on ~1-2% of
  pairs (pair accuracy 1.4, flip rate 1.3-2.4 per seed, ALL), editing only the retrieved text gives almost the whole
  effect (58.4 vs 62.0 for both). The remaining 3.6 points (5.4 on RAG-MISS, positive on every seed) show a small KV
  effect once the text agrees with the edit: the hybrid reads the retrieved text predominantly, the KV facts very little.
  The claim-only selector mask changes nothing (96.3 vs 96.2 acc), so "the selector reads the RAG text" does not
  explain the hybrid either. The hybrid's lead over RAG (ALL: 96.2 vs 91.5 acc, 61.6 vs 50.8 cf) therefore comes mostly
  through the text path. The hybrid's encoder is frozen: its only trainable weights are the KV branch (W_Q/K/V of the KG
  in 7 layers), so the likely mechanism is the KV branch acting as a text-conditioned adapter with the KG states as a
  fixed basis, trained at 50x RAG's learning rate. That is an explanation, not a measurement (see the post-hoc checks).
- RAG-MISS caveat: it is defined on the unedited pass; on a False claim's edited pass the re-indexed retriever can find
  the edited row (its new tail is the claimed one), so RAG keeps 58.3 cf there (50.8 on ALL).
- **K: no branch fires.** `kv_oracle` - `kv_full` = 88.1 - 81.4 = 6.7 (rule: >= 10), `kv_oracle_goldonly` - `kv_oracle`
  = -1.0 (rule: >= 5), `kv_oracle_goldonly` = 87.1 (rule: < 85 for "reader"). The rules did not say what to do when
  none fires (a gap in this registration). Read literally, there is no S2 branch for pure KV. In substance: with
  perfect eval-time routing the KV reader of this recipe (frozen encoder, random distractors in training) reaches
  88.1 acc / 41.4 cf, still below the trained RAG (91.5 / 50.8). A reader trained differently (hard negatives) is not
  ruled out by this. `kv_oracle_goldonly` puts all KG attention on 1 triple for True claims (a setting never seen in
  training), so its lower accuracy is partly an artefact; its cf (45.4; 52.5 on seed 1) is the best pure-KV number.
  UNSEEN-HEAD gives the same picture (kv_full 79.3 / 30.0, kv_oracle 87.0 / 42.2, rag 89.5 / 55.0, hybrid_qclaim_cfkv 1.5).
- **Decision: Stop** (a deviation, see above). No S2 training. The result is negative for this recipe: on FactKG the
  KV memory alone does not beat RAG on accuracy or on faithfulness to an edited fact (also not with oracle routing on
  ALL), and the hybrid's gain over RAG comes mostly through its retrieved text, not through the KG facts in its KV.

## Post-hoc checks (26/11/2026, `run_all.sh factkg_s0b`, 3 seeds each, `results/factkg_s0b`)
| (ALL, 2,327 claims) | acc | cf pair acc |
|---|---|---|
| RAG, 2 unfrozen layers (registered baseline) | 91.5 | 50.8 |
| RAG, 6 unfrozen layers | 96.0 (95.8-96.1) | 60.7 (60.0-61.1) |
| hybrid, real KG | 96.2 (95.7-96.7) | 61.6 (57.3-64.2) |
| hybrid trained with shuffled KV values | 95.6 (95.5-95.8) | 62.8 (62.2-63.8) |

The hybrid with a meaningless KV (values shuffled) matches the real one, and a RAG with more trainable layers matches
both. The hybrid's lead over the registered RAG baseline is adapter capacity, not KG knowledge read from the KV.
No rollback fired in these six runs.
