"""Pick the checkpoint just before a run's collapse (for paired rescue tests, run_all.sh stage `pairs`).

Prints "<ckpt_step> <stop_step>" for a run dir with --probe_every records and ckpt_<step>.pt files: the onset is the first probe
after which every probe shows a collapse (margin < 0.1 and [DEC] cosine at L21 > 0.99); the checkpoint is the last one at least
50 steps before it; stop = onset + 100. Prints nothing if the run did not collapse.
"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
pr = [json.loads(l) for l in (run / "train_log.jsonl").open(encoding="utf-8") if '"probe"' in l]
bad = [p["probe"]["margin_kg"] < 0.1 and p["probe"]["cos_dec_l21"] > 0.99 for p in pr]
onset = next((pr[i]["step"] for i in range(len(pr)) if all(bad[i:])), None) if pr and bad[-1] else None
cks = sorted(int(f.stem.split("_")[1]) for f in run.glob("ckpt_*.pt"))
ck = max((c for c in cks if onset is not None and c <= onset - 50), default=None)
if ck is not None:
    print(ck, onset + 100)
