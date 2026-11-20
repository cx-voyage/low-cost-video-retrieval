"""创新点 B / C：只用 results.csv 里的实测数字做推导，不新增实验。

B. 预算约束下的配置选择
   —— 给定建库时间预算 B，选哪个抽帧率？增量成本换来的精度是多少？

C. 规模变大后哪个成本项主导
   —— 算力成本 vs 存储成本（每帧）；内存墙出现在多少帧。

所有推导只读 results/results.csv，输出 results/derived_budget_scale.csv。
假设（报告中必须写明）：
  机器时间 50 CNY/h（与 §IV 一致），内存 25 CNY/GB（2026 年 DDR4 市场价量级）。
"""

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "results.csv")
OUT = os.path.join(ROOT, "results", "derived_budget_scale.csv")

MACHINE_CNY_PER_H = 50.0
RAM_CNY_PER_GB = 25.0          # 内存（索引驻留内存时用这个价）
SSD_CNY_PER_GB = 0.5           # 固态盘（索引放磁盘时用这个价）
SEC_PER_FRAME = 0.065          # 1 fps 实测 65 ms/帧
BYTES_PER_FRAME = 512 * 4      # float32, 2,048 B
RAM_GB = 15.8                  # 本机内存


def main():
    with open(CSV, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["dataset"] == "msrvtt_1k" and r["agg"] == "max"
                and r["build_seconds"]]
    pts = sorted(({"fps": float(r["fps"]), "sec": float(r["build_seconds"]),
                   "mb": float(r["index_mb"]), "r1": float(r["recall@1"]) * 100,
                   "r10": float(r["recall@10"]) * 100}
                  for r in rows), key=lambda d: d["sec"])

    print("=== B. 预算约束下的配置选择（基准点 = 1.0 fps，见 §V-C）===")
    print(f"{'fps':>5} {'建库(s)':>9} {'R@1':>7} {'R@10':>7} {'vs 上一档':>12} {'s/pp':>9}")
    ref = next(p for p in pts if abs(p["fps"] - 1.0) < 1e-9)
    out = []
    prev = None
    for p in sorted(pts, key=lambda d: d["fps"]):
        inc, rate = "", ""
        if prev is not None:
            dsec = p["sec"] - prev["sec"]
            dpp = p["r1"] - prev["r1"]
            inc = f"{dsec:.0f} s / {dpp:+.1f} pp"
            rate = f"{dsec / dpp:.0f}" if abs(dpp) > 1e-9 else "inf"
            out.append(("B", f"increment_{prev['fps']:g}_to_{p['fps']:g}_fps",
                        f"{dsec:.1f}", "s per 1000 clips"))
            out.append(("B", f"increment_{prev['fps']:g}_to_{p['fps']:g}_fps_pp",
                        f"{dpp:+.1f}", "pp"))
        print(f"{p['fps']:>5g} {p['sec']:>9.1f} {p['r1']:>6.1f}% {p['r10']:>6.1f}% "
              f"{inc:>18} {rate:>8}")
        prev = p

    print(f"\n基准点 1.0 fps：{ref['sec']:.1f} s, R@1 {ref['r1']:.1f}%, R@10 {ref['r10']:.1f}%")
    for p in sorted(pts, key=lambda d: d["fps"]):
        if p is ref:
            continue
        print(f"  {p['fps']:>4g} fps 比基准多 {p['sec'] - ref['sec']:+.1f} s，"
              f"R@1 {p['r1'] - ref['r1']:+.1f} pp，R@10 {p['r10'] - ref['r10']:+.1f} pp")

    out.append(("B", "reference_fps", "1.0", "fps"))
    out.append(("B", "reference_seconds", f"{ref['sec']:.1f}", "s per 1000 clips"))

    print("\n=== C. 算力 vs 存储（每帧）===")
    c_compute = SEC_PER_FRAME * MACHINE_CNY_PER_H / 3600.0
    c_ram = BYTES_PER_FRAME / 1e9 * RAM_CNY_PER_GB
    c_ssd = BYTES_PER_FRAME / 1e9 * SSD_CNY_PER_GB
    print(f"算力成本 {c_compute:.8f} CNY/帧（65 ms × 50 CNY/h）")
    print(f"内存成本 {c_ram:.8f} CNY/帧（2048 B × {RAM_CNY_PER_GB:g} CNY/GB）"
          f" → 算力/内存 = {c_compute / c_ram:.1f}×")
    print(f"磁盘成本 {c_ssd:.8f} CNY/帧（2048 B × {SSD_CNY_PER_GB:g} CNY/GB）"
          f" → 算力/磁盘 = {c_compute / c_ssd:.0f}×")
    parity_price = c_compute / (BYTES_PER_FRAME / 1e9)
    print(f"要让存储与算力持平，存储需卖到 {parity_price:.0f} CNY/GB"
          f"（内存现价假设的 {parity_price / RAM_CNY_PER_GB:.0f} 倍）")

    # 参数化：机时价格 P 变化时谁占主导（50 CNY/h 只是锚点之一）
    k = SEC_PER_FRAME / 3600.0          # 每帧每 CNY/h 的算力成本
    print("\n    机时价格敏感性（内存 25 CNY/GB）：")
    for P, tag in ((0.03, "仅电费"), (1.0, "云 CPU 实例"), (MACHINE_CNY_PER_H, "本报告锚点")):
        c = k * P
        verdict = (f"存储贵 {c_ram / c:5.1f}×" if c_ram > c else f"算力贵 {c / c_ram:5.1f}×")
        print(f"      P = {P:5.2f} CNY/h ({tag:<8})  {verdict}")
    print(f"    → 交叉点 P* = {c_ram / k:.2f} CNY/h（内存） / {c_ssd / k:.4f} CNY/h（SSD）")
    print("      低于 P* 存储更贵，高于 P* 算力更贵；比值与 N 无关。")

    n_wall = RAM_GB * 1e9 / BYTES_PER_FRAME
    print(f"\n内存墙：{RAM_GB} GB / {BYTES_PER_FRAME} B = {n_wall:.2e} 帧 "
          f"= {n_wall / 3600:.0f} 小时视频（1 fps）")
    print(f"到内存墙时的建库算力：{n_wall * SEC_PER_FRAME / 3600:.0f} h "
          f"= {n_wall * SEC_PER_FRAME / 3600 * MACHINE_CNY_PER_H:.0f} CNY")

    out += [
        ("C", "compute_cny_per_frame", f"{c_compute:.8f}", "CNY/frame"),
        ("C", "storage_cny_per_frame_ram", f"{c_ram:.8f}", "CNY/frame"),
        ("C", "storage_cny_per_frame_ssd", f"{c_ssd:.8f}", "CNY/frame"),
        ("C", "compute_over_storage_ram", f"{c_compute / c_ram:.2f}", "ratio"),
        ("C", "compute_over_storage_ssd", f"{c_compute / c_ssd:.0f}", "ratio"),
        ("C", "storage_price_parity", f"{parity_price:.0f}", "CNY/GB"),
        ("C", "ram_wall_frames", f"{n_wall:.0f}", "frames"),
        ("C", "ram_wall_hours_of_video", f"{n_wall / 3600:.0f}", "hours"),
        ("C", "build_hours_at_ram_wall", f"{n_wall * SEC_PER_FRAME / 3600:.0f}", "h"),
        ("C", "build_cny_at_ram_wall",
         f"{n_wall * SEC_PER_FRAME / 3600 * MACHINE_CNY_PER_H:.0f}", "CNY"),
        # 参数化写法：机时价格 P 是变量，50 CNY/h 只是其中一个锚点
        ("C", "compute_cny_per_frame_basis", f"{MACHINE_CNY_PER_H:g}", "CNY/h"),
        ("C", "compute_cny_per_frame_per_cny_per_hour",
         f"{SEC_PER_FRAME / 3600:.10f}", "CNY/frame per CNY/h"),
        ("C", "crossover_machine_price_ram",
         f"{c_ram / (SEC_PER_FRAME / 3600):.2f}", "CNY/h"),
        ("C", "crossover_machine_price_ssd",
         f"{c_ssd / (SEC_PER_FRAME / 3600):.4f}", "CNY/h"),
        ("C", "compute_over_storage_ram_at_0.03cny_per_h",
         f"{k * 0.03 / c_ram:.4f}", "ratio"),
        ("C", "storage_over_compute_ram_at_0.03cny_per_h",
         f"{c_ram / (k * 0.03):.1f}", "ratio"),
        ("C", "compute_over_storage_ram_at_1cny_per_h",
         f"{k * 1.0 / c_ram:.3f}", "ratio"),
    ]

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "item", "value", "unit"])
        w.writerows(out)
    print(f"\n[已保存] {OUT}")


if __name__ == "__main__":
    main()
