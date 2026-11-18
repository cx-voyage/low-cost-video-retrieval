"""图 2：成本-精度曲线（报告里最重要的一张图）。

数据只从 results/results.csv 读，不手填任何数字。

用法：
  py code\make_figure.py

输出：
  results/fig_cost_accuracy.png   （300 dpi，3.5 英寸宽 = IEEE 单栏）
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "results.csv")
OUT = os.path.join(ROOT, "results", "fig_cost_accuracy.png")

# 统计口径：n=1000 时 R@1 的标准误 ≈ 1.5 个百分点
SE_PP = 1.5


def load():
    rows = []
    with open(CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "msrvtt_1k" or r["agg"] != "max":
                continue
            if not r["build_seconds"]:
                continue
            rows.append({
                "fps": float(r["fps"]),
                "n_frames": int(r["n_frames"]),
                "sec": float(r["build_seconds"]),
                "mb": float(r["index_mb"]),
                "r1": float(r["recall@1"]) * 100,
                "r5": float(r["recall@5"]) * 100,
                "r10": float(r["recall@10"]) * 100,
            })
    rows.sort(key=lambda d: d["sec"])
    return rows


def main():
    rows = load()
    if not rows:
        raise SystemExit("results.csv 里没有 msrvtt_1k / max 的行")

    plt.rcParams.update({
        "font.size": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = [r["sec"] for r in rows]

    # 主指标 R@1
    y1 = [r["r1"] for r in rows]
    ax.plot(x, y1, "o-", color="#1f4e79", lw=1.2, ms=4,
            label="Recall@1", zorder=3)

    # 1 个标准误的带子：带内差异不可解读
    base = next((r["r1"] for r in rows if abs(r["fps"] - 1.0) < 1e-9), y1[0])
    ax.axhspan(base - SE_PP, base + SE_PP, color="#1f4e79", alpha=0.12,
               lw=0, zorder=1)
    ax.text(x[0] * 1.12, base + SE_PP + 0.5, r"$\pm$1 s.e. (1.5 pp)",
            fontsize=7, color="#1f4e79")

    # 参考指标 R@10
    ax.plot(x, [r["r10"] for r in rows], "s--", color="#a6611a", lw=1.0,
            ms=3.2, label="Recall@10", zorder=2)

    # 每个点标 fps
    for r in rows:
        ax.annotate(f"{r['fps']:g} fps", (r["sec"], r["r1"]),
                    textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=7, color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Indexing time per 1,000 clips (s, log scale)")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(25, 72)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f} s" for v in x])
    # 关掉对数轴的次刻度标签，否则会和自定义刻度叠在一起
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, loc="center right", fontsize=7)

    # 标出拐点
    knee = next((r for r in rows if abs(r["fps"] - 1.0) < 1e-9), None)
    if knee:
        ax.annotate("selected\noperating point", (knee["sec"], knee["r1"]),
                    textcoords="offset points", xytext=(6, 10),
                    fontsize=7, color="#1f4e79",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#1f4e79"))

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=300)
    print(f"[已生成] {OUT}")
    print("\n数据点：")
    print(f"  {'fps':>5} {'帧数':>7} {'建库(s)':>9} {'索引(MB)':>9} "
          f"{'R@1':>6} {'R@5':>6} {'R@10':>6}")
    for r in rows:
        print(f"  {r['fps']:>5g} {r['n_frames']:>7} {r['sec']:>9.1f} "
              f"{r['mb']:>9.2f} {r['r1']:>6.1f} {r['r5']:>6.1f} {r['r10']:>6.1f}")
    lo, hi = rows[0], rows[-1]
    print(f"\n成本跨度 {hi['sec'] / lo['sec']:.1f}x"
          f"（{lo['sec']:.1f}s -> {hi['sec']:.1f}s），"
          f"R@1 跨度 {max(r['r1'] for r in rows) - min(r['r1'] for r in rows):.1f} pp")


if __name__ == "__main__":
    main()
