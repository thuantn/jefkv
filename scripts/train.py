"""
Train JefKV (or a RAG / hybrid baseline with the same budget) on data/processed/train.json.

Schedule follows AtlasKV (Huang et al., ICLR 2026, Appendix A.1 & E): AdamW, cosine LR, sub-KG
curriculum 10 -> 120 triples (+4 every 100 steps), --random_kv for the RandomKV check.

Modes
  kv      KG memory in attention only (trainable: KG heads + temperature)
  rag     retrieved triple sentences in the state (trainable: top --unfreeze_top encoder layers)
  hybrid  both (trainable: KG heads + temperature)
"""

import argparse
import io
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from jefkv.memory import DenseRetriever, KGMemory, minilm_encoder, triple_text
from jefkv.model import JefKVModel


def amp_dtype(device: str, amp: str) -> torch.dtype:
    if amp == "auto":  # JEFKV_AMP=fp16 on a bf16-capable GPU keeps the T4 recipe (every published run is fp16)
        amp = os.environ.get("JEFKV_AMP", "auto")
    if device != "cuda" or amp == "fp32":
        return torch.float32
    if amp == "auto":  # T4 (sm75) has no real bf16
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]


def build_memory(model: JefKVModel, triples, encoder: str, device: str, dtype, max_slots: int, store: str = "") -> KGMemory:
    if encoder == "minilm":  # faithful AtlasKV KG2KV baseline
        return KGMemory(triples, minilm_encoder(device), device, dtype, atlaskv_kv=True)
    enc = lambda texts, max_len=max_slots: model.encode_kg_texts(texts, max_len)
    return KGMemory(triples, enc, store or device, dtype)  # store: where the memory lives (cpu: only batches go to the GPU)


def rag_contexts(retriever, samples, k):
    return [" ".join(triple_text(retriever.triples[i]) for i in retriever.search(s["state"], k)[0]) for s in samples]


def decision_loss(logits, labels, present):
    """Per-sample mean: CE when the rule exists, KL to 50/50 when it is missing."""
    lp = F.log_softmax(logits, -1)
    ce = -lp.gather(1, labels[:, None]).squeeze(1)
    kl = (0.5 * (torch.log(torch.tensor(0.5, device=lp.device)) - lp)).sum(-1)
    return torch.where(present, ce, kl).mean()


PROBE_COS_LAYERS, PROBE_RATIO_LAYERS = (11, 21), (6, 18, 21)


@torch.no_grad()
def probe(model: JefKVModel, packed, labels, kg, device: str, dt) -> dict:
    """Diagnostics on a fixed batch of rule-present training cases (no grad, no RNG, so training is unchanged):
    |margin| and accuracy with and without the KG, mean pairwise cosine of the [DEC] state across cases
    (layers 11 and 21; 1.00 = every case looks the same), median ||y_kg|| / ||y_seq|| at [DEC] (KV layers 6, 18, 21)
    and the mean temperature factor."""
    rows = torch.arange(labels.shape[0], device=labels.device)
    dec = packed.dec_pos  # one question per case: dec_pos[b] is case b's [DEC] position
    hidden, layers = {}, model.jef.encoder.layers
    hooks = [layers[i].register_forward_hook(lambda m, inp, out, i=i: hidden.__setitem__(i, out[0] if isinstance(out, tuple) else out))
             for i in PROBE_COS_LAYERS]
    kvm = {i: model.kg_modules[i] for i in PROBE_RATIO_LAYERS if i in model.kg_modules}
    for m in kvm.values():
        m.debug = {}
    amp = torch.autocast(device_type=device, dtype=dt, enabled=(dt != torch.float32))
    try:
        with amp:
            lg = model(packed, kg)[0].float()
        out = {"temp": model.last_temp.float().mean().item()}
        for i, h in hidden.items():
            x = F.normalize(h[rows, dec].float(), dim=-1)
            c = x @ x.T
            out[f"cos_dec_l{i}"] = ((c.sum() - c.trace()) / (len(x) * (len(x) - 1))).item()
        for i, m in kvm.items():
            ys = m.debug["y_seq"][rows, :, dec].flatten(1).norm(dim=-1)
            yk = m.debug["y_kg"][rows, :, dec].flatten(1).norm(dim=-1)
            out[f"kg_seq_l{i}"] = (yk / ys.clamp(min=1e-8)).median().item()
    finally:
        for h in hooks:
            h.remove()
        for m in kvm.values():
            m.debug = None
    with amp:
        lg0 = model(packed, None)[0].float()
    for tag, x in (("kg", lg), ("nokg", lg0)):
        d = x[:, 1] - x[:, 0]
        out[f"margin_{tag}"] = d.abs().mean().item()
        out[f"acc_{tag}"] = ((d > 0).long() == labels).float().mean().item()
    return out


def retrieval_loss(triple_logits, batch, sub_ids):
    """Mean of -log P(g) over every gold triple g (a multi-rule case needs BOTH rules), per layer."""
    pos = {g: j for j, g in enumerate(sub_ids)}
    pairs = [(b, pos[g]) for b, s in enumerate(batch) for g in s["gold_idx"]]
    if not pairs or not triple_logits:
        return None
    b_idx = torch.tensor([b for b, _ in pairs], device=next(iter(triple_logits.values())).device)
    g_idx = torch.tensor([g for _, g in pairs], device=b_idx.device)
    losses = []
    for tl in triple_logits.values():
        ok = tl[b_idx, g_idx] > -1e30  # gold among the triples this layer attended (top-k routing)
        if ok.any():
            losses.append(-F.log_softmax(tl, -1)[b_idx[ok], g_idx[ok]].mean())
    return torch.stack(losses).mean() if losses else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/train.json")
    p.add_argument("--mode", choices=["kv", "rag", "hybrid"], default="kv")
    p.add_argument("--kv_encoder", choices=["jef", "minilm"], default="jef")
    p.add_argument("--random_kv", action="store_true")
    p.add_argument("--save_dir", default="runs/kv_s0")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--enc_lr", type=float, default=2e-5, help="LR for unfrozen encoder layers (rag).")
    p.add_argument("--min_lr_ratio", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200, help="linear LR warmup steps, all modes (~7%% of 3000 steps, not tuned; 0 = none, as in gate 7)")
    p.add_argument("--unfreeze_top", type=int, default=-1, help="-1 = auto: 2 for rag, 0 otherwise.")
    p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--rag_k", type=int, default=3)
    p.add_argument("--max_slots", type=int, default=16)
    p.add_argument("--kg_store", default="", help="cpu = keep the training KG memory in RAM, move each batch's subset to the GPU")
    p.add_argument("--init_kg_size", type=int, default=4)  # AtlasKV App A.1: 4, +4 / 100 steps, max 120
    p.add_argument("--kg_step_inc", type=int, default=4)
    p.add_argument("--max_kg_size", type=int, default=120)
    p.add_argument("--amp", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze_kg_v", action="store_true", help="keep W_V^kg at its init (the layer's own W_V)")
    p.add_argument("--restart_on_collapse", type=int, default=0,
                   help="max restarts (new seed) when the run is stuck at 50/50 at --collapse_check; 0 = off")
    p.add_argument("--probe_every", type=int, default=0,
                   help="every N steps log probe() on a fixed batch of 32 training cases (0 = off)")
    p.add_argument("--ckpt_at", type=int, nargs="*", default=[],
                   help="also save adapter + optimizer + data-order RNG at these steps (ckpt_<step>.pt)")
    p.add_argument("--stop_at", type=int, default=0, help="stop after this step (LR schedule still over --steps); no adapter.pt")
    p.add_argument("--resume", default="", help="ckpt_<step>.pt to continue from (adapter, optimizer, data-order RNG)")
    p.add_argument("--reseed", type=int, default=None, help="with --resume: new data-order seed from the resumed step")
    p.add_argument("--perturb", type=float, default=0.0, help="with --resume: relative Gaussian noise on the adapter (chaos control)")
    p.add_argument("--stop_on_collapse", type=int, default=0, help="stop after N probes in a row show a collapse (needs --probe_every)")
    p.add_argument("--step_log", action="store_true", help="log batch make-up, grad / update norms per group and the scale every step")
    p.add_argument("--rollback_on_collapse", type=int, default=0,
                   help="max rollbacks: on a collapse (4 probes in a row), go back to an in-RAM snapshot >=150 steps earlier, "
                        "add relative noise 1e-4 to the adapter and continue with the same batches (kv / hybrid; sets --probe_every 25)")
    p.add_argument("--collapse_check", type=int, default=1500)
    p.add_argument("--collapse_margin", type=float, default=0.1,
                   help="collapsed if the EMA of |logit margin| on rule-present training cases is below this")
    args = p.parse_args()
    if args.unfreeze_top < 0:
        args.unfreeze_top = 2 if args.mode == "rag" else 0
    if args.rollback_on_collapse and not args.probe_every:
        args.probe_every = 25
    args.seed_requested, args.restarts, args.collapsed_at_check = args.seed, 0, None
    while not train_once(args):
        # Frozen-Jef runs sometimes fall into an "always 50/50" solution early and never leave it (README,
        # "Training collapse"). It shows on training batches only (no eval data), so restart with a new seed.
        args.restarts += 1
        args.seed = args.seed_requested + 1000 * args.restarts
        print(f"collapsed at step {args.collapse_check}: restart {args.restarts} with seed {args.seed}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_once(args) -> bool:
    """One training run; False if --restart_on_collapse is on, restarts remain and the run collapsed."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = amp_dtype(device, args.amp)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    samples, triples = data["samples"], data["triples"]

    use_kv = args.mode in ("kv", "hybrid")
    kg_dim = 384 if args.kv_encoder == "minilm" else 768
    model = JefKVModel(kg_embed_dim=kg_dim, use_kv=use_kv, unfreeze_top=args.unfreeze_top, device=device,
                       freeze_kg_v=args.freeze_kg_v)

    memory = None
    if use_kv:
        memory = build_memory(model, triples, args.kv_encoder, device, dt if device == "cuda" else torch.float32, args.max_slots, args.kg_store)
        if args.random_kv:
            memory.shuffle_values(args.seed)
    retriever = DenseRetriever(triples, device) if args.mode in ("rag", "hybrid") else None

    # Encoder weights (top layers and final_norm when unfrozen) at enc_lr; final_norm used to get lr (50x).
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("jef.encoder.layers", "jef.encoder.final_norm"))
                  and ".attn.kg_" not in n]
    enc_ids = {id(p) for p in enc_params}
    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups = [g for g in ({"params": new_params, "lr": args.lr}, {"params": enc_params, "lr": args.enc_lr}) if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=(dt == torch.float16))
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"mode={args.mode} kv_encoder={args.kv_encoder} random_kv={args.random_kv} trainable={n_train:,} amp={dt}")

    probe_batch = None
    if args.probe_every and memory is not None:
        prng = random.Random(4242)  # its own RNG: probing must not change the training batches
        ps = prng.sample([s for s in samples if s["rule_present"] and s["gold_idx"]], 32)
        pgold = sorted({g for s in ps for g in s["gold_idx"]})
        pids = pgold + prng.sample([i for i in range(len(memory)) if i not in set(pgold)], 120)
        probe_batch = (model.batch(ps, rag_contexts(retriever, ps, args.rag_k) if retriever else None),
                       torch.tensor([s["label"] for s in ps], device=device), torch.tensor(pids, device=device))

    def snapshot(step):  # everything a resume needs (checkpoints and in-RAM rollback snapshots)
        return {"args": vars(args), "step": step, "state": model.trainable_state(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng.getstate(), "ema": ema, "ema_margin": ema_margin}

    def restore(ck):
        model.load_trainable(ck["state"])
        base = [g["base_lr"] for g in opt.param_groups]
        opt.load_state_dict(ck["opt"])
        for g, b in zip(opt.param_groups, base):  # --lr / --enc_lr of this call, not the checkpoint's
            g["base_lr"] = b
        scaler.load_state_dict(ck["scaler"])
        rng.setstate(ck["rng"])
        return ck["step"], ck.get("ema"), ck.get("ema_margin")

    def perturb(rel, seed):  # relative Gaussian noise on every trainable weight
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in new_params + enc_params:
                p.mul_(1 + rel * torch.randn(p.shape, generator=gen).to(p.device))

    step, ema, ema_margin = 0, None, None
    if args.resume:  # rescue tests: continue a run from a checkpoint, optionally with one factor changed
        step, ema, ema_margin = restore(torch.load(args.resume, map_location=device, weights_only=False))
        if args.reseed is not None:
            rng = random.Random(args.reseed)
        if args.perturb:
            perturb(args.perturb, 0)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    def pgroup(n):  # parameter groups for --step_log (read-only diagnostics)
        for key, g in (("kg_v_proj", "v"), ("kg_q_proj", "qk"), ("kg_k_proj", "qk"), ("kg_gate", "gate"), ("kg_prior", "prior"), ("temp.", "temp")):
            if key in n:
                return g
        return "other"
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    model.train()
    t0, n_bad, n_rb_bad, n_rollbacks, snaps = time.time(), 0, 0, 0, {}
    bar = tqdm(total=args.steps, initial=step, desc=save.name)
    with (save / "train_log.jsonl").open("a" if args.restarts else "w", encoding="utf-8") as log:  # restarts append
        while step < args.steps:
            if args.rollback_on_collapse and (step % 100 == 0 or not snaps):  # in RAM (CPU), the last 3 only
                buf = io.BytesIO()
                torch.save(snapshot(step), buf)
                snaps = {s: b for s, b in snaps.items() if s > step - 300} | {step: buf}
            step += 1
            bar.update(1)
            frac = 0.5 * (1 + math.cos(math.pi * (step - 1) / max(1, args.steps - 1)))
            # Linear warmup: damps Adam's first steps (each moves all 11M adapter weights by ~lr). It changed which
            # seeds collapse to 50/50 but did not stop collapses (README, "Training collapse").
            warm = min(1.0, step / args.warmup) if args.warmup > 0 else 1.0
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * warm * (args.min_lr_ratio + (1 - args.min_lr_ratio) * frac)

            batch = rng.choices(samples, k=args.batch_size)
            kg, sub_ids = None, None
            if memory is not None:
                # Curriculum size = number of distractors (always added on top of the batch's gold
                # triples, so retrieval is never trivial even when a batch has many gold rules).
                if args.unfreeze_top and args.kv_encoder == "jef" and step % 100 == 1 and (step > 1 or n_rollbacks):  # after a rollback to 0 too
                    # Trainable top layers change the states the KG keys are made of: re-encode so the
                    # keys seen in training match the ones eval builds from the trained weights.
                    memory = kg = None  # free the old KG first: two full memories at once OOM on FactKG
                    memory = build_memory(model, triples, args.kv_encoder, device, dt if device == "cuda" else torch.float32, args.max_slots, args.kg_store)
                    if args.random_kv:
                        memory.shuffle_values(args.seed)
                size = min(args.max_kg_size, args.init_kg_size + (step // 100) * args.kg_step_inc)
                gold = {g for s in batch for g in s["gold_idx"]}
                pool = [i for i in range(len(memory)) if i not in gold]
                sub_ids = list(gold) + rng.sample(pool, min(size, len(pool)))
                rng.shuffle(sub_ids)
                kg = tuple(x.to(device) for x in memory.subset(torch.tensor(sub_ids, device=memory.device)))
            ctx = rag_contexts(retriever, batch, args.rag_k) if retriever else None
            packed = model.batch(batch, ctx)
            labels = torch.tensor([s["label"] for s in batch], device=device)
            present = torch.tensor([s["rule_present"] for s in batch], device=device)

            with torch.autocast(device_type=device, dtype=dt, enabled=(dt != torch.float32)):
                logits, tl, _ = model(packed, kg)
            dec = decision_loss(logits, labels, present)
            # Frozen scorers (selecting layer: Q/K at identity) get no gradient from the aux loss;
            # keeping them would only dilute the other layers' terms.
            # With routing, only the selecting layer retrieves; aux on later layers taught them to match the
            # triple's NAME (gate 5 probe: entity -> head <=45%, tail <=2%), so they learn from the decision only.
            tl = {i: x for i, x in tl.items() if model.kg_modules[i].kg_q_proj.weight.requires_grad
                  and (model.kg_modules[i].select_k or not any(m.select_k for m in model.kg_modules.values()))}
            aux = retrieval_loss(tl, batch, sub_ids) if sub_ids else None
            loss = dec + args.aux_weight * aux if aux is not None else dec

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.step_log:  # measured only: nothing below changes the update
                sl, before = {"present": int(present.sum()), "label1": int(labels.sum()), "scale": scaler.get_scale()}, {}
                for n, p in named:
                    g = pgroup(n)
                    if p.grad is not None:
                        sl["g_" + g] = sl.get("g_" + g, 0.0) + p.grad.float().pow(2).sum().item()
                    before[n] = p.detach().clone()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            if args.step_log:
                for n, p in named:
                    g = pgroup(n)
                    sl["u_" + g] = sl.get("u_" + g, 0.0) + (p.detach() - before[n]).float().pow(2).sum().item()
                    sl["w_" + g] = sl.get("w_" + g, 0.0) + before[n].float().pow(2).sum().item()
                sl = {k: (v ** 0.5 if k[:2] in ("g_", "u_", "w_") else v) for k, v in sl.items()}
                sl["skipped"] = scaler.get_scale() < sl["scale"]
                log.write(json.dumps({"step": step, "loss_step": loss.item(), "dec_step": dec.item(), "sl": sl}) + "\n")

            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
            if present.any():
                m = (logits[present, 1] - logits[present, 0]).float().abs().mean().item()
                ema_margin = m if ema_margin is None else 0.98 * ema_margin + 0.02 * m
            if step == args.collapse_check:  # recorded even without restarts, so tables can count collapses
                args.collapsed_at_check = ema_margin is not None and ema_margin < args.collapse_margin
                if args.collapsed_at_check and args.restarts < args.restart_on_collapse:
                    log.write(json.dumps({"step": step, "collapsed": True, "ema_margin": ema_margin}) + "\n")
                    return False
            if step % 20 == 0 or step == 1:
                rec = {"step": step, "loss": loss.item(), "ema_loss": ema, "dec_loss": dec.item(), "ema_margin": ema_margin,
                       "aux_loss": None if aux is None else aux.item(), "kg_size": None if kg is None else len(sub_ids),
                       "random_kv": args.random_kv, "secs": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if probe_batch is not None and (step % args.probe_every == 0 or step == 1):
                pk, pl, pids = probe_batch
                pr = probe(model, pk, pl, tuple(x.to(device) for x in memory.subset(pids.to(memory.device))), device, dt)
                log.write(json.dumps({"step": step, "probe": pr}) + "\n")
                log.flush()
                # collapsed on the probe batch (margin ~0, [DEC] case-independent) for stop_on_collapse probes in a row
                n_bad = n_bad + 1 if pr["margin_kg"] < 0.1 and pr["cos_dec_l21"] > 0.99 else 0
                if args.stop_on_collapse and n_bad >= args.stop_on_collapse:
                    return True
                # Rollback criterion: the one above missed a collapse whose [DEC] cosine stayed 0.93-0.99 (randomkv s2),
                # and margin alone misfires early (healthy runs dip to 0.05 before step ~300). On every probe log so far
                # (5 collapses, 20 runs without one) this one flags each collapse 75-100 steps after onset and nothing else.
                n_rb_bad = n_rb_bad + 1 if pr["margin_kg"] < 0.1 and (pr["cos_dec_l21"] > 0.99 or pr["margin_kg"] < 0.01) else 0
                if args.rollback_on_collapse and n_rb_bad >= 4 and n_rollbacks < args.rollback_on_collapse:
                    # Collapses are knife-edge events (README, "Training collapse"): a 1e-4 nudge avoided 3/3 from just
                    # before onset, with the same batches. Go back >=150 steps (onset is up to ~100 steps before detection).
                    to = max((s for s in snaps if s <= step - 150), default=min(snaps))
                    n_rollbacks += 1
                    log.write(json.dumps({"step": step, "rollback_to": to, "rollback": n_rollbacks}) + "\n")
                    step, ema, ema_margin = restore(torch.load(io.BytesIO(snaps[to].getvalue()), map_location=device, weights_only=False))
                    snaps = {s: b for s, b in snaps.items() if s <= to}
                    perturb(1e-4, n_rollbacks)
                    n_bad = n_rb_bad = 0
                    bar.n = step
                    bar.refresh()
                    continue
            if step in args.ckpt_at:
                torch.save(snapshot(step), save / f"ckpt_{step}.pt")
            if step == args.stop_at:
                return True

    torch.save({"args": vars(args), "state": model.trainable_state()}, save / "adapter.pt")
    print(f"saved {save / 'adapter.pt'} after {time.time() - t0:.0f}s")
    return True


if __name__ == "__main__":
    main()
