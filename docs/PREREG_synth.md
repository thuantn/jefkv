# Synthetic benchmark: capacity checks (registered 26/11/2026, before running; revised after two reviews, same day)

**Question.** On the synthetic benchmark (`data/processed/test.json`) frozen-Jef KV beats the best RAG: OOD 85.2 / cf
58.9 (`kv_jef`, `results/rb_check`, 6 seeds, the recommended recipe) vs 79.0 / 44.6 (RAG, BM25 k=3, no warmup, 3 seeds);
the hybrid reaches 97.2 / 94.3 (rb_check, 6 seeds). (The README's 84.6 / 59.4 is the earlier recipe on seeds 0-5.)
On FactKG the same kind of lead disappeared once RAG trained more layers (RAG6 = hybrid = hybrid with shuffled KV,
`docs/PREREG_factkg.md`). RAG here trains its top 2 encoder layers (10.0M params) at LR 2e-5, the KV heads 11.2M at
1e-3: close in size, not in learning rate or place in the network. Does the KV lead survive (a) a RAG with 6 trainable
layers and (b) the comparison at matched trainable encoder layers? Does the hybrid use the facts in its KV?

**Step 0, eval only** (`run_all.sh synth_diag`, the existing rb_check hybrid adapters s0-2): `hybrid_qclaim` (selector
scores with the claim tokens only), and its cf with the edit reaching only the KV (`_cfkv`) or only the retrieved text
(`_cfrag`). Until now the synthetic cf edited both paths at once.

**Runs** (`run_all.sh synth_cap`, 3 seeds each, all with `--rollback_on_collapse 3`, every run reported):
- `rag_unf6_nowarm_s{0,1,2}`: `--mode rag --unfreeze_top 6 --warmup 0` (RAG's recipe with 6 layers instead of 2).
- `kv_jef_unf6_rbv_s{0,1,2}`: `--mode kv --unfreeze_top 6` (KV with the same 6 trainable encoder layers, plus its heads).
- `hybrid_randomkv_rbv_s{0,1,2}`: `--mode hybrid --random_kv` (hybrid, KV values shuffled; keys still hold the tail).

**What is compared.** Declared now: RAG rows use **BM25 k=3** (the README's best-RAG recipe); every other eval row is
reported too. OOD acc and cf pair, means over the non-collapsed seeds; `multi`, `alias`, trap reported, not decided on.
A collapsed run (OOD within 50 +- 3) is reported and left out of the means; 2 of 3 collapsed = no verdict for that row.

**Decision rules.**
1. *Plug-in KV lead* (frozen KV vs RAG6). Holds if `kv_jef` (85.2 / 58.9) is >= 2.0 OOD **and** >= 5.0 cf above RAG6.
   Otherwise the synthetic lead of frozen KV over RAG is written up as trainable capacity, as on FactKG.
2. *Matched lead* (`kv_jef_unf6` vs RAG6, same trainable encoder layers; KV also has its 11.2M heads, so a KV win is still
   not fully matched, a loss is). Same margins as rule 1.
3. *Hybrid uses the KG*, two readings that must agree: (a) `hybrid_qclaim_cfkv` pair accuracy >= 20 (an edit of the KV
   alone changes the answer; FactKG: 2.1); (b) the shuffled hybrid's cf is >= 10 below the real hybrid (94.3). Both
   hold = the hybrid reads its KV facts; neither = the KV branch works as an adapter on the retrieved text; one only =
   reported as mixed.

**Not tested here.** A RAG with a higher encoder LR (the 50x LR difference stays a confound if rule 1 fails); new data.
