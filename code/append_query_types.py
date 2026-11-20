"""把创新点 A 的分层结果写进 results/results.csv（追加，不改动已有行）。"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "results.csv")
SRC = os.path.join(ROOT, "results", "query_types.csv")

with open(SRC, encoding="utf-8", newline="") as f:
    strat = list(csv.DictReader(f))

with open(CSV, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())

existing = {(r["dataset"], r["fps"]) for r in rows}
added = 0
for s in strat:
    ds = f"msrvtt_1k_qtype_{s['category']}"
    if (ds, s["fps"]) in existing:
        continue
    row = {c: "" for c in cols}
    row.update({
        "date": "2026-11-18",
        "dataset": ds,
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "fps": s["fps"],
        "agg": "max",
        "max_frames": "30",
        "n_clips": "1000",
        "recall@1": s["recall@1"],
        "recall@5": s["recall@5"],
        "recall@10": s["recall@10"],
        "rand@1": "0.001",
        "rand@5": "0.005",
        "rand@10": "0.01",
        "note": (f"query-type slice n={s['n_queries']} (rule-based labels, see "
                 f"query_types.csv); same index as msrvtt_1k"),
    })
    rows.append(row)
    added += 1

with open(CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"appended {added} rows -> {CSV} (total {len(rows)})")
