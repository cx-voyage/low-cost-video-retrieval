"""把一次评测结果汇总成 results.csv 的一行。

为什么需要这个脚本：
  evaluate.py --save 只保存"本次评测的 JSON"（Recall、配置）。
  但报告用的 results.csv 是一张汇总表，字段还包括建库耗时、索引体积、帧数，
  这些不在 eval JSON 里。以前需要人工抄，容易漏（2.0 fps 就是这么漏的）。
  本脚本把 eval JSON + index 目录里的 config.json 自动拼成一行，并追加到 results.csv。

用法：
  python code/record_result.py --eval results/msrvtt_vitb32_fps1.5.json \
      --index data/index_msrvtt_fps1.5 --build-seconds 1759.8

  --build-seconds 省略时，会去读 index 目录 config.json 里的 build_seconds
  （新版 build_index.py 会自动写入；旧索引没有，就必须显式给）。

行为：
  - 幂等：若 results.csv 已有 (dataset, fps, agg, n_frames) 相同的行，直接跳过。
  - 只追加，不覆盖 -> 符合实验记录规范。
"""
import argparse
import csv
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "results.csv")

HEADER = ["date", "dataset", "model", "pretrained", "fps", "agg", "max_frames",
          "n_clips", "n_frames", "build_seconds", "ms_per_frame", "index_mb",
          "recall@1", "recall@5", "recall@10",
          "rand@1", "rand@5", "rand@10",
          "text_encode_ms", "search_ms", "note"]

DATASET_NAME = {"msrvtt": "msrvtt_1k", "ucf": "ucf101_subset"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="evaluate.py --save 产生的 JSON")
    ap.add_argument("--index", required=True, help="索引目录，如 data/index_msrvtt_fps1.5")
    ap.add_argument("--build-seconds", type=float, default=None,
                    help="建库总耗时（秒）；省略则读 index/config.json 的 build_seconds")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k",
                    help="权重短名，默认 laion2b_s34b_b79k")
    ap.add_argument("--text-encode-ms", default="")
    ap.add_argument("--search-ms", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ev = json.load(open(args.eval, encoding="utf-8"))
    idir = args.index if os.path.isabs(args.index) else os.path.join(ROOT, args.index)
    icfg = json.load(open(os.path.join(idir, "config.json"), encoding="utf-8"))

    protocol = ev["protocol"]
    dataset = DATASET_NAME.get(protocol, protocol)
    cfg = ev.get("config", {})

    fps = cfg.get("fps", icfg.get("fps"))
    n_frames = cfg.get("n_frames", icfg.get("n_frames"))
    n_clips = cfg.get("n_clips", icfg.get("n_clips"))
    model = cfg.get("model", icfg.get("model"))
    max_frames = cfg.get("max_frames", icfg.get("max_frames"))

    if n_frames != icfg.get("n_frames"):
        print(f"[警告] eval 记录 {n_frames} 帧，索引实际 {icfg.get('n_frames')} 帧 —— "
              f"请确认 --eval 与 --index 是同一配置", file=sys.stderr)
    if abs(float(fps) - float(icfg.get("fps", fps))) > 1e-9:
        print(f"[警告] eval fps={fps} 与索引 fps={icfg.get('fps')} 不一致", file=sys.stderr)

    build_seconds = args.build_seconds
    if build_seconds is None:
        build_seconds = icfg.get("build_seconds")
    if build_seconds is None:
        print("[错误] 不知道建库耗时。旧索引没有记录，请用 --build-seconds 指定"
              "（从 log.md 里抄那次运行的总耗时）。", file=sys.stderr)
        sys.exit(2)

    emb_path = os.path.join(idir, "emb.npy")
    index_mb = os.path.getsize(emb_path) / 1e6 if os.path.exists(emb_path) else ""
    ms_per_frame = round(float(build_seconds) * 1000 / max(int(n_frames), 1), 1)

    row = {
        "date": args.date, "dataset": dataset, "model": model,
        "pretrained": args.pretrained, "fps": fps, "agg": ev["agg"],
        "max_frames": max_frames, "n_clips": n_clips, "n_frames": n_frames,
        "build_seconds": build_seconds, "ms_per_frame": ms_per_frame,
        "index_mb": round(index_mb, 2) if index_mb != "" else "",
        "recall@1": ev.get("Recall@1"), "recall@5": ev.get("Recall@5"),
        "recall@10": ev.get("Recall@10"),
        "rand@1": ev.get("random@1"), "rand@5": ev.get("random@5"),
        "rand@10": ev.get("random@10"),
        "text_encode_ms": args.text_encode_ms, "search_ms": args.search_ms,
        "note": args.note,
    }

    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    key = lambda r: (r["dataset"], float(r["fps"]), r["agg"], str(r["n_frames"]))
    for r in rows:
        if key(r) == key(row):
            print(f"[跳过] results.csv 已有 {dataset} fps={fps} agg={ev['agg']} "
                  f"frames={n_frames} 的行，未重复写入。")
            return

    if args.dry_run:
        print("[dry-run] 将要追加：")
        print(", ".join(f"{h}={row[h]}" for h in HEADER))
        return

    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(f"[已追加] results.csv -> {dataset} | {fps} fps | {ev['agg']} | "
          f"{n_frames} 帧 | {build_seconds} s | {ms_per_frame} ms/帧 | "
          f"{row['index_mb']} MB | "
          f"R@1={ev.get('Recall@1')} R@5={ev.get('Recall@5')} R@10={ev.get('Recall@10')}")
    print(f"现在共 {len(rows) + 1} 行。")


if __name__ == "__main__":
    main()
