"""README figures (light + dark SVG): accuracy and counterfactual pair accuracy.
results_*: synthetic benchmark, numbers from README "Results" (rb_check: the recommended recipe, rollback on collapse,
every seed counted; synth_cap: the capacity checks). factkg_*: FactKG test, 3 seeds, numbers from docs/PREREG_factkg.md (S0 + post-hoc checks).
  python scripts/plot_summary.py docs/"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name, family, (acc mean, std), (cf pair mean, std) or None
FIGS = {
    "results": dict(
        rows=[
            ("Hybrid: RAG text + KV memory", "kv", (97.2, 0.7), (94.3, 1.3)),
            ("Hybrid, KV values shuffled", "ctrl", (96.1, 0.4), (92.8, 1.0)),
            ("RAG, 6 trainable layers (BM25, k=3)", "rag", (89.4, 1.8), (71.7, 3.9)),
            ("KV memory + 6 trainable layers", "kv", (86.0, 0.5), (61.2, 1.2)),
            ("KV memory, frozen Jef (JefKV)", "kv", (85.2, 0.5), (58.9, 2.1)),
            ("RAG, 2 trainable layers (BM25, k=3)", "rag", (79.0, 1.0), (44.6, 0.6)),
            ("RandomKV control (shuffled facts)", "ctrl", (64.4, 1.1), (7.9, 3.2)),
            ("Zero-shot Jef (no knowledge)", "ctrl", (51.9, 0.0), None),
        ],
        acc="Synthetic benchmark: OOD accuracy (%)",
        note="Mean ± std over seeds, every run counted (3-6). JefKV beats a RAG that trains 2 encoder layers, not one that\n"
             "trains 6, even when KV trains the same 6; a shuffled KV does as well as the real one inside the hybrid."),
    "factkg": dict(
        rows=[
            ("Hybrid: RAG text + KV memory", "kv", (96.2, 0.4), (61.6, 3.1)),
            ("Hybrid, KV values shuffled", "ctrl", (95.6, 0.1), (62.8, 0.7)),
            ("RAG, 6 trainable layers", "rag", (96.0, 0.1), (60.7, 0.5)),
            ("RAG, 2 trainable layers", "rag", (91.5, 0.3), (50.8, 0.6)),
            ("KV memory, frozen Jef (JefKV)", "kv", (81.4, 0.2), (28.0, 1.0)),
            ("String-match rule (no training)", "ctrl", (77.1, 0.0), None),
            ("RandomKV control", "ctrl", (71.7, 0.6), (7.8, 1.4)),
            ("Zero-shot Jef", "ctrl", (55.0, 0.0), None),
        ],
        acc="FactKG (real DBpedia claims): accuracy (%)",
        note="Mean ± std over 3 seeds, 2,327 test claims. Negative result: a shuffled KV does as well as the real one,\n"
             "and a RAG with more trainable layers matches the hybrid. JefKV alone stays below RAG."),
}
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  kv="#2a78d6", rag="#eb6834", ctrl="#898781"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 kv="#3987e5", rag="#d95926", ctrl="#898781"),
}
CF = "Counterfactual pair accuracy (%)\n= right both before and after the fact is edited"


def draw(fig_def, theme, path):
    t, rows = THEMES[theme], fig_def["rows"]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 1.6 + 0.42 * len(rows)), sharey=True, facecolor=t["surface"])
    ys = list(range(len(rows)))[::-1]
    for ax, (title, col, ref) in zip(axes, [(fig_def["acc"], 2, "chance"), (CF, 3, None)]):
        ax.set_facecolor(t["surface"])
        for y, row in zip(ys, rows):
            val = row[col]
            if val is None:
                ax.text(1, y, "n/a (reads no KG / not edited)", va="center", color=t["muted"], fontsize=9)
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
    axes[0].set_yticks(ys, [r[0] for r in rows], color=t["ink"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[k]) for k in ("kv", "rag", "ctrl")]
    fig.legend(handles, ["KV memory in attention", "RAG (retrieved text)", "controls"], loc="lower center", ncol=3,
               frameon=False, fontsize=9, labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.09))
    fig.text(0.5, 0.012, fig_def["note"], color=t["muted"], fontsize=8, ha="center")
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.savefig(path, facecolor=t["surface"], metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/"
    for name, fd in FIGS.items():
        for th in THEMES:
            draw(fd, th, f"{out.rstrip('/')}/{name}_{th}.svg")
