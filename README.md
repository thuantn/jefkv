# JefKV: In-Attention Knowledge Graph Memory for a Calibrated Decision Model

**JefKV** is a personal research proof-of-concept. It asks whether **AtlasKV-style parametric KG memory** (Huang et al., ICLR 2026) helps a compact, bidirectional **System-1 decision model**, [Decision-Jef-0.1](https://huggingface.co/BarraHome/Decision-Jef-0.1) (307M, mmBERT-base / ModernBERT), decide rule-dependent cases, and where it does better or worse than plain retrieval-augmented prompting (RAG). It is an independent project, not affiliated with or endorsed by the authors of Decision-Jef or AtlasKV.

> **Status:** 3-seed results for every configuration and seeds 3–5 for the frozen-Jef ones (T4). Two fixes for the training collapse and a re-evaluation on a fresh test set are running; this README will be updated with them. Numbers in "Changes" are from development gate runs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results_dark.svg">
  <img alt="OOD accuracy: hybrid 97.2, JefKV 84.6, best RAG 79.0, RandomKV 65.4, zero-shot 51.9. Counterfactual pair accuracy: hybrid 94.5, JefKV 59.4, best RAG 44.6, RandomKV 8.5." src="docs/results_light.svg">
</picture>

## What is combined

- **Decision-Jef-0.1** answers typed questions in one forward pass and reads each answer at a `[DEC]` token with its own heads. JefKV keeps that decision path and Jef's heads **frozen**, so Jef's calibration is not thrown away.
- **AtlasKV / KBLaM** inject knowledge-graph triples as extra key/value columns into attention, mixed with a separated softmax (JefKV replaces it with a bounded gate, see "Changes"). JefKV wraps Jef's **global** attention layers (3, 6, …, 21). ModernBERT puts global attention every 3 layers, which is also AtlasKV's best injection interval (K=3, Appendix B.3).

## Research questions

1. **Retrieval inside a bidirectional encoder.** Does separated-softmax KV memory work inside a frozen ModernBERT, and which global layer retrieves? Checked with the RandomKV control (AtlasKV Appendix E) and per-layer ACC@1/5.
2. **KV memory vs. RAG, on the same budget.** Which wins, and on which kind of case: single rule, rules that must be **combined**, entity names written in a **different surface form**, and **missing** rules?
3. **Knowing when the rule is missing.** When no rule matches, does the model fall back towards 50%, or stay confidently wrong? Measured as confidence on the trap split and the abstention AUROC (confidence separating rule-present from rule-missing cases).

## Changes from the AtlasKV recipe (and why)

| Change | Why |
|---|---|
| **Token-level, layer-matched KV from the backbone itself.** Each triple sentence is encoded by frozen Jef; for each KV layer, the token states (≤16) that this layer's attention sees become its memory slots. W_Q^kg = W_K^kg = identity, W_V^kg = the layer's own W_V (AtlasKV inits W_Q^kg from W_Q). MiniLM sentence vectors (AtlasKV/KBLaM style) are kept as an ablation (`--kv_encoder minilm`). | One sentence vector blurs exact numbers and IDs; token states keep them. Gate run 3 used final-layer states for every layer: queries of layer i did not match them (untrained OOD ACC@5 10–57% vs 95–98% layer-matched), and with W_Q-copied/random K the KG learned only the relation type (ACC@5 ≈ 0 on the 10K test KG). With identity init, layer 3 finds the gold triple (OOD ACC@5 100% among 160) before any training. |
| **Auxiliary retrieval loss** on a ColBERT-style triple score: each case token takes its best slot of the triple (max), averaged over case tokens; −log softmax of the gold triple, per layer. **Inactive in the final setup:** with top-16 routing only the selecting layer (3) retrieves, and its Q/K are frozen, so no layer receives this loss; the later layers learn from the decision loss only (the code path stays for runs without routing). | The benchmark knows which triple is relevant, so retrieval can be supervised directly. Averaging tokens first (gate run 3) scored triples against the whole case, i.e. by relation type: the 2–3 entity tokens were drowned. Eval reports this score's ACC@5 (`score_acc@5`) next to AtlasKV's post-softmax ACC@k. |
| **Retrieve-then-attend inside the model (top-16 routing).** The first KV layer (3) scores every triple (ColBERT score above) and keeps the top 16 per case (its Q/K frozen at identity: training them made it a worse retriever on the 10K test KG, OOD ACC@5 97.5% -> 77.5% in gate run 4); every KV layer attends only to those (~256 slots instead of ~161K), and the later layers re-rank them (auxiliary loss over the 16 candidates). | Gate run 4: retrieval was right (ACC@5 95% on the 10K KG) but attention over all slots let almost none of the right triple's value through: decisions 64% vs 78% for trained RAG, at 710 ms vs 45 ms per case. Layer 3 is the best untrained retriever (OOD ACC@5 97.5% among 2K triples). AtlasKV's HiKVP also keeps a top-k of triples. |
| **One-level GPU K-Means pruning** instead of HiKVP's 3-level UMAP + GMM hierarchy (Sec 4.2, App B.4). | At ≤10K triples the whole memory fits in VRAM; pruning is reported only as a latency/accuracy trade-off. |
| **KG scores in fp32, soft-capped at 50·tanh(x/50)** (Gemma 2 style). | First gate run: the retrieval loss drove KG scores to ~8,000, fp16 overflowed (NaN), and KG attention swamped the sequence so decisions collapsed to 50/50. |
| **Gated mixing instead of the separated softmax**: out = (1−g)·softmax(seq)·V + g·softmax(KG)·V_kg, with g = 0.5·sigmoid(w) per head (init 0.15; g = 0 when no triple is attended). | In the joint softmax the KG's share grows with the number of slots: gate run 2 collapsed to 50/50 (λ_kg ≈ 1), and after a −log(#slots) fix the trained adapter (gate 5) still gave the KG 63–98% of the mass, with [DEC]/number tokens putting 80–94% of it on distractors. KV then ignored the tail (counterfactual pair acc 1.3% ≈ RandomKV). A bounded gate keeps the case readable and makes the KG share independent of KG size. |
| **Triple prior**: the selecting layer's z-scored triple scores (over the top 16) × a learned scalar (init 3) are added to the KG logits of every KV layer. | Later layers re-rank the 16 candidates from their own (weaker) scores; the prior lets them start from the selector's ranking. At init it cut the distractor share of KG mass to 11–16% and sent 27–40% of a number token's KG mass to the gold tail. |
| **Linear LR warmup, 200 steps, for every configuration (KV, RandomKV, hybrid, RAG)** (`--warmup`, default 200; AtlasKV uses cosine decay only). **200 is not tuned**: it is the usual 5–10% of training (200/3000 ≈ 7%), the only value tried, and the same for all baselines so no method gets a tuned schedule. | Added against a training collapse (below), but **it does not remove it**. Without warmup, 1 seed in 3 of every frozen-Jef configuration (KV, RandomKV, hybrid) ended at 50/50 on every split; the collapse is deterministic per seed (the seed only sets the batch order, Jef has no dropout: KV seed 2 gave 49.3% twice). Warmup rescued those three seeds (KV s2 49.3 → 81.1%, hybrid s0 54.0 → 97.6%, RandomKV s1 50.7 → 67.5%), but with warmup two *other* seeds collapsed (hybrid s1, RandomKV s2), so it changes which seeds fail, not whether runs fail. Kept because it is harmless (KV 0/6 failures with it, 2/18 frozen-Jef runs over seeds 0–5, RAG −1 to −2 pts, both RAG recipes are reported). Too short a warmup may not damp the first Adam steps; too long a one eats the high-LR part of a 3000-step cosine schedule while the KG curriculum grows, risking under-training. |
| **Retrieval-conditioned temperature**: logits × softplus(w·f + b), f = per-layer [λ_kg, max triple probability]. | Lets the model flatten towards 50% when nothing in the KG matches (RQ3). RAG runs get a learned global temperature only. |

## Baselines (all on the same frozen Jef)

| Name | Knowledge source | Trainable |
|---|---|---|
| `zs_base` | none (Jef zero-shot) | – |
| `zs_rag` | top-3 retrieved triple sentences in the state (dense MiniLM or BM25) | – |
| `rag` | same, evaluated at k = 1, 3, 10 and with BM25 | top 2 encoder layers (10.0M, vs 11.2M for the KV heads) |
| `kv_jef` | full KG in attention (all test triples, or K-Means-pruned on GPU), top-16 routing | KV heads + temperature (11.2M) |
| `kv_jef_unf2` | as `kv_jef` | KV heads + temperature **+ top 2 encoder layers** (21.2M) — diagnostic, see below |
| `kv_minilm` | AtlasKV's own KG2KV: MiniLM key `"the <relation> of <head>"`, value = tail only | KV heads + temperature |
| `hybrid` | RAG context **and** KV memory | KV heads + temperature |
| `kv_jef_randomkv`, `kv_jef_unf2_randomkv` | keys paired with random values (control) | as `kv_jef` / `kv_jef_unf2` |

**`kv_jef_unf2` is a diagnostic, not the main configuration.** Gate run 5: KV ≈ RandomKV (~61%, a prior learnable without knowledge) while trained RAG reached 78%, and RAG also trains the top 2 encoder layers. `unf2` tests whether KV only lacked trainable layers to learn the number comparison. What it trades away:
1. **No longer plug-in** (the main cost): Jef itself changes, so without a KG the model is no longer stock Jef (the pass-through self-test does not hold for this adapter) and Jef's calibration may shift. AtlasKV keeps the LLM frozen.
2. **Unequal budget vs RAG**: 21.2M vs 10.0M trainable. KV vs RandomKV stays fair (same 21.2M); vs RAG the gap is reported.
3. **KG depends on trained weights**: layer-21 keys come from layer 20's output, so training re-encodes the KG every 100 steps (keys up to 100 steps stale) and any model update means re-encoding the whole KG.
4. **Slower training** (backprop through 2 layers + ~30 re-encodes); inference latency unchanged.
5. **More room to learn template shortcuts** from 15K synthetic cases; RandomKV with the same layers measures that part.

If it helps, the results table reports both rows (frozen = AtlasKV's plug-in premise; `unf2` = matched to trained RAG) with these trade-offs.

**Counterfactual tail-swap (`cf` in eval.json, `ood` split).** Plain accuracy can be reached without reading the KG: RandomKV learns a case-only prior of ~61–65%. For each OOD case the gold triple's tail is replaced, in memory (KV) or in the retrieved text (RAG), by a value that flips the correct answer. `pair_acc` = both the original and the swapped case right (≈ 0 for any model that ignores the tail); `flip_rate` = the prediction changes. It proves the tail is read and used, not that the comparison with the case value is exact.

Note on budgets: `kv_full` attends over the *whole* KG (10K triples) while RAG sees only k retrieved sentences. That is the premise of in-attention memory, not an oracle: the model still has to find the right triple, and per-layer ACC@1/5 show whether it does. RAG is also run at k = 10.

## Results (3 seeds, T4)

Mean ± std over seeds 0–2 on the held-out splits (1,000 OOD cases, 500 per other split). `cf pair` = counterfactual tail-swap pair accuracy on OOD (≈ 0 if the KG value is ignored). "collapsed" = runs that ended at ~50% confidence on every split (see below). Warmup 200 unless stated.

| run | collapsed | OOD acc | multi | alias | cf pair | trap conf (→50) | abstain AUROC | ms/case |
|---|---|---|---|---|---|---|---|---|
| zero-shot Jef | – | 51.9 | 50.0 | 49.2 | – | 88.5 | 0.48 | 21 |
| zero-shot Jef + RAG (dense k=3 / BM25 k=3) | – | 51.4 / 51.7 | 49.0 / 52.0 | 49.4 / 48.8 | 0.8 / 1.9 | 94 | 0.48 | 28 / 39 |
| `rag` dense k=3, no warmup | 0/3 | 78.0 ± 1.5 | **62.0 ± 2.4** | 66.9 ± 1.4 | 43.1 ± 1.1 | 71.8 | 0.55 | **29** |
| `rag` BM25 k=3, no warmup (best RAG) | 0/3 | 79.0 ± 1.0 | 58.1 ± 1.1 | 64.1 ± 2.4 | 44.6 ± 0.6 | 67.0 | 0.64 | 44 |
| `rag` dense k=3 | 0/3 | 76.0 ± 2.3 | 60.8 ± 2.1 | **67.2 ± 0.9** | 40.4 ± 2.7 | 70.4 | 0.56 | 30 |
| `rag` BM25 k=3 | 0/3 | 77.9 ± 1.2 | 56.6 ± 0.9 | 64.7 ± 0.6 | 42.7 ± 1.3 | 64.9 | 0.65 | 41 |
| `rag` dense k=10 | 0/3 | 65.6 ± 0.8 | 57.0 ± 1.4 | 63.7 ± 0.8 | 14.2 ± 0.1 | 69.8 | 0.52 | 30 |
| `kv_jef_randomkv` (control) | 1/3 | 60.3 ± 10.9 | 56.8 ± 7.1 | 59.7 ± 6.2 | 10.5 ± 1.9 | 58.8 | 0.52 | 116 |
| `kv_minilm` (AtlasKV KG2KV, 1 seed) | – | 59.9 | 52.8 | 62.2 | 0.0 | 59.7 | 0.50 | 44 |
| **`kv_jef`** (frozen Jef, plug-in) | 0/3 | **84.4 ± 3.2** | **64.2 ± 3.1** | 65.9 ± 1.3 | **58.1 ± 5.5** | 67.2 | 0.68 | 115 |
| `kv_jef_unf2` (diagnostic, not plug-in) | 0/3 | 86.6 ± 0.5 | 66.1 ± 2.0 | 67.9 ± 0.7 | 63.0 ± 3.0 | 70.1 | 0.71 | 113 |
| `hybrid` (RAG text + KV); in brackets: the 2 runs that did not collapse | 1/3 | 82.4 ± 25.6 (97.6, 96.7) | 61.7 ± 8.7 (67.4, 66.0) | 64.9 ± 11.2 (71.4, 71.4) | 63.6 ± 53.1 (95.2, 93.3) | 61.1 (66.0) | 0.79 (0.95) | 175 |

**Seeds 3–5** (frozen-Jef configurations only, free Colab T4, same recipe): none of the 9 runs collapsed. OOD acc / cf pair per seed: `kv_jef` 85.2 / 59.4, 84.5 / 62.2, 84.8 / 60.6; `kv_jef_randomkv` 64.6 / 8.3, 64.9 / 9.3, 64.4 / 5.8; `hybrid` 97.6 / 95.2, 98.1 / 95.8, 96.2 / 93.1. Over seeds 0–5, `kv_jef` is 84.6 ± 2.0 OOD (cf 59.4 ± 3.9, multi 65.3 ± 2.4, alias 66.5 ± 1.8) and the 5 `hybrid` runs that trained are 97.2 ± 0.8 (cf 94.5 ± 1.2).

What this says:
1. **The frozen-Jef KV memory uses the knowledge.** cf pair 58% vs 10% for RandomKV and 0% for AtlasKV's own MiniLM KG2KV (which only learns the ~60% case-only prior).
2. **KV vs RAG (compared with RAG's better recipe):** KV is +5.4 pts on OOD (84.4 vs 79.0) and reads the retrieved fact more reliably (cf 58 vs 45); on rules that must be combined (`multi`) it is +2.2 over the best RAG row on that split (dense k=3, no warmup). RAG stays better on alias names (67.2 vs 65.9), as expected from a text retriever, and is ~3–4× faster (30–44 vs 115 ms per case).
3. **Both together (`hybrid`) is the strongest when it trains** (97% OOD, 94% cf) but collapsed in 1 of 3 runs in each recipe on seeds 0–2 (none of seeds 3–5).
4. **Missing rules (RQ3):** no method reaches the ideal 50% confidence on `trap` cases; KV-based models separate rule-present from rule-missing cases better (abstention AUROC 0.68–0.71 for KV, 0.95 for the hybrid runs that trained, vs 0.55–0.65 for RAG).
5. **Pruning:** the K-Means-pruned KV variant (top 8 of 128 clusters on pooled keys) drops the gold triple before routing (layer-3 ACC@5 27% vs 88%): OOD 61.8%, i.e. no knowledge. It halves latency but is not usable as built.

**Training collapse (open problem).** With Jef frozen, some runs fall into an "always 50/50" solution and never leave it: the train decision loss reaches ≈ 0.59 (= 0.85·ln 2) by step ~200 and the logit margin on rule-present cases then decays to ~0 between steps 500 and 1,200.
- *What it looks like:* in a collapsed adapter the [DEC] hidden state is the same for every case from layer 11 on (pairwise cosine 1.00), and the KG read-out is 5–12× the sequence read-out in the last KV layers (`scripts/probe_drown.py`).
- *Where it lives:* swapping parameter groups between a good and a collapsed KV adapter (no-warmup seeds 1 and 2, 40 OOD cases, `scripts/probe_swap.py`) puts it in the KG value projections `W_V^kg`. The good adapter with the collapsed `W_V^kg` falls to 30% (margin 0.16); the collapsed adapter with the good `W_V^kg` recovers to 87%. Swapping the gate, prior or temperature changes nothing.
- *How often:* without warmup, 3 of 9 frozen-Jef runs (seeds 0–2) collapsed; with warmup 200, 2 of 18 (seeds 0–5: KV 0/6, RandomKV 1/6, hybrid 1/6); with the top 2 Jef layers trainable, 0/6. It is deterministic per seed.
- *Workaround, restart on collapse* (`--restart_on_collapse 2`): if an EMA of the train margin is below 0.1 at step 1,500, the run restarts with seed + 1000. Both collapsed warmup seeds were caught (EMA 0.014 and 0.002) and retrained into normal runs: hybrid s1 52.9 → 97.5 OOD (cf 94.9), RandomKV s2 47.8 → 66.8 (cf 7.6). The tables above report runs *without* restarts.
- *Tried and dropped:* capping the KG read-out at the sequence's RMS removed the collapse mechanism but cost ~11 pts (gate run 8). Two single-factor fixes are being screened: a lower LR (3e-4) and freezing `W_V^kg` at its initialisation (`--freeze_kg_v`).

Per-seed numbers are in each `runs/<config>_s<seed>/eval.json`; `bash run_all.sh full` also writes means to `outputs/results.md` and plots. The no-warmup RAG rows come from `rag_nowarm_s*` (same stage).

## Limitations (read before citing a number)

- **No separate validation split.** The design decisions in "Changes" (gate runs 1–9) were made by looking at the test split, so the KV numbers above are optimistic for this design. A re-evaluation of the final adapters on a fresh test set (new entities, different generator seed) is reported below when available.
- **Synthetic, single-family task.** One kind of rule (is the observed number within an entity's limit), 4 domains, English pseudo-word entities. Values at the edges of a domain's range give the label without any knowledge, which is why models with no access to the KG reach ~60% (RandomKV, MiniLM KG2KV) and why the counterfactual `cf pair` metric is reported.
- **RandomKV is a conservative control.** It shuffles values only (AtlasKV App. E); its keys still contain each triple's own tail tokens, so a weak key-side channel remains (its cf pair is 10%, not 0%). This can only make the control stronger, i.e. shrink KV's measured advantage.
- **Training exposure differs.** KV training batches always contain the gold triples plus a growing set of distractors (AtlasKV's curriculum); RAG trains on what its retriever returns (dense k=3, ~97% gold recall on OOD). At test time both use the same full 10K-triple KG.
- **cf asymmetry.** For RAG the edited tail replaces the gold sentence only if the unedited index retrieved it; for KV the edited triple is re-encoded in memory and re-scored. The edit only changes the tail number, which is not in the query, so retrieval is expected to be essentially unchanged.
- **One backbone, small seeds count.** 3 seeds per configuration (plus seeds 3–5 for the frozen-Jef ones); 2 of 18 frozen-Jef runs with warmup collapsed (see above).

## Benchmark (synthetic, `data/generate_benchmark.py`)

Four policy domains (return window, expense limit, warranty, baggage). Each case asks whether an observed value is within the entity's limit; the limit exists only in the KG. Entity names are pronounceable pseudo-words; train, test and missing-rule names are disjoint, and nothing in the text reveals whether a rule exists.

| Split | Cases | What it tests |
|---|---|---|
| `ood` | 1,000 | single rule, 10K unseen test entities |
| `multi` | 500 | needs the entity limit **plus** a customer-tier bonus rule (the observed value always exceeds the base limit) |
| `alias` | 500 | entity written differently from the KG (`velorakin`, `Ve Lo Ra Kin`, `VE-LO-RA-KIN`) |
| `trap` | 500 | entity has no rule; ideal confidence ≈ 50% |

Limits: synthetic English data, numeric threshold rules only; Jef itself is English-only.

## Project structure

```text
jefkv/attention.py      separated-softmax KG attention for ModernBERT global layers (transformers 5.x)
jefkv/memory.py         KG memory (Jef token states or MiniLM), GPU K-Means pruning, dense + BM25 retrievers
jefkv/model.py          Decision-Jef wrapper: KV injection, frozen [DEC] heads, retrieval-conditioned temperature
data/generate_benchmark.py
scripts/train.py        modes kv | rag | hybrid, --random_kv, --kv_encoder, AtlasKV curriculum, fp16 on T4
scripts/eval.py         accuracy, ECE, trap confidence, abstention AUROC, per-layer ACC@1/5, latency
scripts/plot_results.py results.md (mean ± std over seeds) + plots
tests/test_jefkv.py     self-check: no KG == stock Jef; masked KG == stock Jef; live KG changes output
run_all.sh              smoke | gate | full
kaggle/run_kaggle.py    entry point for a Kaggle GPU kernel
```

## Running

```bash
pip install -r requirements.txt          # pinned: transformers 5.24.0, decision-jef 0.8.1
PYTHONPATH=. python tests/test_jefkv.py  # ~1 min on CPU once weights are cached
bash run_all.sh smoke                    # every code path on tiny data, CPU
bash run_all.sh gate                     # step 1 on a GPU: clean KV vs RandomKV (seed 0)
bash run_all.sh full                     # all configs x 3 seeds + zero-shot baselines -> outputs/results.md
```

`gate` is a stop/go check: if the clean-KG loss does not separate from the RandomKV loss, the KV heads are not retrieving and the rest is not worth running. On a T4 training runs in fp16 automatically. Other stages: `warm` (the warmup check above), `seeds` (seeds 3–5 of the frozen-Jef configs), `restart` (restart on collapse), `screen_lr` / `screen_fkv` (the two collapse fixes), `fresh` (re-evaluation on a fresh test set), `unf`, `rag`.

Where to get a GPU (Modal and Colab ran this code on a T4, Kaggle ran the CPU `smoke` stage; one KV run ≈ 10 min incl. eval, `full` ≈ 3–4 h):

| Option | Cost | How |
|---|---|---|
| Colab (free T4) | free, subject to Colab's usage limits | Colab CLI (Linux/macOS/WSL): `uv tool install google-colab-cli`, then `bash colab/run_stage.sh <stage>`; results in `runs_colab/<stage>/` |
| Kaggle (free T4) | free, the GPU queue can take hours | `KAGGLE_ENV=<file with KAGGLE_API_TOKEN=...> bash kaggle/push.sh <stage> new` (first time; `version` afterwards); results via `kaggle kernels output <user>/jefkv-run` |
| Modal (T4) | ~$0.25 per run, ~$3.5 for `full` | `modal deploy run_modal.py && python run_modal.py <stage>` (keeps running if your machine disconnects; `modal run --detach` did not), results on the `jefkv-runs` volume, then `modal app stop jefkv` |

## References & Acknowledgments

This project builds directly upon the following papers, architectures, and open-source releases:

1. **AtlasKV (Core Parametric KG Memory Method):** Haoyu Huang, Hong Ting Tsang, Jiaxin Bai, Xi Peng, Gong Zhang, and Yangqiu Song. *AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM*. Published at **ICLR 2026** ([arXiv:2510.17934](https://arxiv.org/abs/2510.17934) | [Official GitHub](https://github.com/HKUST-KnowComp/AtlasKV)).
2. **KBLaM (Rectangular Attention Paradigm):** Xi Wang, Taketomo Isazawa, Liana Mikaelyan, and James Hensman. *KBLaM: Knowledge Base Augmented Language Model*. 2024 ([arXiv:2410.10450](https://arxiv.org/abs/2410.10450)).
3. **Decision-Jef-0.1 (System-1 Calibrated Decision Model):** `BarraHome`. *Decision-Jef-0.1* (307M calibrated decision model). 2026 ([Hugging Face](https://huggingface.co/BarraHome/Decision-Jef-0.1)).
4. **mmBERT (Multilingual Encoder Backbone):** Marc Marone, Orion Weller, William Fleshman, Eugene Yang, Dawn Lawrie, and Benjamin Van Durme. *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning*. 2025 ([arXiv:2509.06888](https://arxiv.org/abs/2509.06888) | [GitHub](https://github.com/jhu-clsp/mmBERT) | [Hugging Face](https://huggingface.co/jhu-clsp/mmBERT-base)).
5. **ModernBERT (Alternating Local/Global Attention Architecture):** Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Griffin Thomas Adams, Jeremy Howard, and Iacopo Poli. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*. **ACL 2025**, pp. 2526–2547 ([ACL Anthology](https://aclanthology.org/2025.acl-long.127/) | [arXiv:2412.13663](https://arxiv.org/abs/2412.13663)).
6. **Sentence-BERT & MiniLM (Default Lightweight Sentence Encoder):**
   - Nils Reimers and Iryna Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. 2019 ([arXiv:1908.10084](https://arxiv.org/abs/1908.10084)).
   - Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. **NeurIPS 2020** ([Hugging Face: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)).
7. **Core Foundations (Transformer & AdamW):**
   - Ashish Vaswani et al. *Attention is All You Need*. **NeurIPS 2017**.
   - Ilya Loshchilov and Frank Hutter. *Decoupled Weight Decay Regularization*. 2017 ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)).

### BibTeX

```bibtex
@inproceedings{huang2026atlaskv,
  title     = {AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM},
  author    = {Huang, Haoyu and Tsang, Hong Ting and Bai, Jiaxin and Peng, Xi and Zhang, Gong and Song, Yangqiu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2510.17934},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2510.17934}
}

@article{wang2024kblam,
  title   = {KBLaM: Knowledge Base Augmented Language Model},
  author  = {Wang, Xi and Isazawa, Taketomo and Mikaelyan, Liana and Hensman, James},
  journal = {arXiv preprint arXiv:2410.10450},
  year    = {2024},
  url     = {https://arxiv.org/abs/2410.10450}
}

@article{marone2025mmbert,
  title   = {mmBERT: A Modern Multilingual Encoder with Annealed Language Learning},
  author  = {Marone, Marc and Weller, Orion and Fleshman, William and Yang, Eugene and Lawrie, Dawn and Van Durme, Benjamin},
  journal = {arXiv preprint arXiv:2509.06888},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.06888}
}

@inproceedings{warner2025modernbert,
  title     = {Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
  author    = {Warner, Benjamin and Chaffin, Antoine and Clavi{\'e}, Benjamin and Weller, Orion and Hallstr{\"o}m, Oskar and Taghadouini, Said and Gallagher, Alexis and Biswas, Raja and Ladhak, Faisal and Aarsen, Tom and Adams, Griffin Thomas and Howard, Jeremy and Poli, Iacopo},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {2526--2547},
  year      = {2025},
  url       = {https://aclanthology.org/2025.acl-long.127/}
}

@misc{barrahome2026decisionjef,
  title        = {Decision-Jef-0.1: Calibrated System-1 Decision Model},
  author       = {{BarraHome}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/BarraHome/Decision-Jef-0.1}}
}

@inproceedings{wang2020minilm,
  title     = {MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers},
  author    = {Wang, Wenhui and Wei, Furu and Dong, Li and Bao, Hangbo and Yang, Nan and Zhou, Ming},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {5776--5788},
  year      = {2020}
}

@article{reimers2019sentencebert,
  title   = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author  = {Reimers, Nils and Gurevych, Iryna},
  journal = {arXiv preprint arXiv:1908.10084},
  year    = {2019},
  url     = {https://arxiv.org/abs/1908.10084}
}

@article{loshchilov2017decoupled,
  title   = {Decoupled Weight Decay Regularization},
  author  = {Loshchilov, Ilya and Hutter, Frank},
  journal = {arXiv preprint arXiv:1711.05101},
  year    = {2017}
}

@inproceedings{vaswani2017attention,
  title     = {Attention is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {30},
  year      = {2017}
}
```

## How this was built (AI assistance)

Code, experiments and this README were written with Claude Opus 5.5 (Claude Code). Design decisions and code were also brainstormed and cross-reviewed by other models, each given the measurements and code, not the conclusions:

| Model | Used for |
|---|---|
| Gemini 4 Neon High | brainstorming at decision points, code reviews (found 4 real bugs in the gate-5 code), cross-review of merged plans (e.g. a "0/14 ≈ 1% collapse" claim was wrong: Fisher p ≈ 0.14 against the baseline) |
| Gemini 4 Argon High | code review |
| Gemini 3.8 Flash Low | checking the implementation against the AtlasKV paper (curriculum, KG2KV key/value format, ACC@k) |
| GPT-6 Luna | brainstorming and cross-review (e.g. screening the collapse fixes one factor at a time, as `screen_lr` / `screen_fkv` do, instead of combining them) |
| DeepSeek V4.1 Flash | code reviews (e.g. an infinite gradient from a clamp placed after a `sqrt` in the dropped RMS cap) |

The gated mixing + triple prior design (gate run 7) came out of one such brainstorm. Every finding was checked and re-run before use, and several suggestions were rejected after checking. All numbers in this README come from the runs, not from the models.

## License

MIT — see [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff) for full attribution details.

Third-party weights and packages are downloaded at run time, not redistributed here (no weights, adapters or vendored code in this repo). Decision-Jef-0.1, its base model `jhu-clsp/mmBERT-base` and the `decision-jef` package are MIT; `sentence-transformers/all-MiniLM-L6-v2` is Apache-2.0; the Python dependencies are MIT / Apache-2.0 / BSD-3 / PSF. AtlasKV is re-implemented from the paper (no code taken from its repository). Decision-Jef's model card notes that part of its training data (`tasksource/tasksource-jev-typed-decisions`) carries `license: other` and that this has not been reviewed by its author. The AtlasKV paper is linked, not included.
