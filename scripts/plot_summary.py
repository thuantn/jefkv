"""Headline figure for the README (light + dark SVG): OOD accuracy and counterfactual pair accuracy.
Numbers are copied from README "Results" (seeds 0-5 for frozen-Jef configs, 0-2 for RAG).
hybrid and RandomKV: the runs that did not collapse (5 of 6 each).
  python scripts/plot_summary.py docs/"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name, family, (OOD mean, std), (cf pair mean, std) or None
ROWS = [
    ("Hybrid: RAG text + KV memory", "kv", (97.2, 0.8), (94.5, 1.2)),
    ("KV memory, frozen Jef (JefKV)", "kv", (84.6, 2.0), (59.4, 3.9)),
    ("Best RAG (BM25, k=3, no warmup)", "rag", (79.0, 1.0), (44.6, 0.6)),
    ("RandomKV control (shuffled facts)", "ctrl", (65.4, 1.3), (8.5, 1.6)),
    ("Zero-shot Jef (no knowledge)", "ctrl", (51.9, 0.0), None),
]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  kv="#2a78d6", rag="#eb6834", ctrl="#898781"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 kv="#3987e5", rag="#d95926", ctrl="#898781"),
}
PANELS = [("OOD accuracy (%)", 2, "chance"), ("Counterfactual pair accuracy (%)\n= right both before and after the fact is edited", 3, None)]


def draw(theme, path):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True, facecolor=t["surface"])
    ys = list(range(len(ROWS)))[::-1]
    for ax, (title, col, ref) in zip(axes, PANELS):
        ax.set_facecolor(t["surface"])
        for y, row in zip(ys, ROWS):
            val = row[col]
            if val is None:
                ax.text(1, y, "n/a (reads no KG)", va="center", color=t["muted"], fontsize=9)
                continue
            m, sd = val
            ax.barh(y, m, height=0.5, color=t[row[1]], zorder=2)
            label = f"{m:.1f}" + (f" ± {sd:.1f}" if sd else "")
            ax.text(m + 1.2, y, label, va="center", color=t["ink"], fontsize=9)
        if ref:
            ax.axvline(50, color=t["muted"], lw=1, zorder=1)
            ax.text(50.8, ys[0] + 0.45, ref, color=t["muted"], fontsize=8, va="bottom")
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=t["grid"], lw=1, zorder=0)
        ax.set_title(title, loc="left", color=t["ink"], fontsize=10)
        ax.tick_params(colors=t["ink2"], length=0, labelsize=9)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
    axes[0].set_yticks(ys, [r[0] for r in ROWS], color=t["ink"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[k]) for k in ("kv", "rag", "ctrl")]
    fig.legend(handles, ["KV memory in attention", "RAG (retrieved text)", "controls"], loc="lower center", ncol=3,
               frameon=False, fontsize=9, labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.012, "Mean ± std over seeds (KV 6, RAG 3; hybrid and RandomKV: the 5 runs that did not collapse).",
             color=t["muted"], fontsize=8, ha="center")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path, facecolor=t["surface"], metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/"
    for th in THEMES:
        draw(th, f"{out.rstrip('/')}/results_{th}.svg")
