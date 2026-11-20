"""配对显著性检验：所有抽帧率两两比较（R@1 与 R@10），McNemar 连续校正。

为什么需要这个脚本
------------------
正文引用了 5 个 p 值（§V-C 的 p=0.16 / 0.03 / 0.88 / 0.11，Contributions 的 p=0.12）。
这些必须能从仓库里一条命令复现，而不是散落在临时脚本里。

做法：对同一条查询，比较两种配置“是否命中”，构造 2x2 列联表，
用 McNemar 检验（b01 与 b10 为不一致对，连续校正），p = erfc(sqrt(chi2/2))。

输出：results/significance.csv
用法：python code/analyze_significance.py
"""

import csv
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from evaluate import aggregate  # noqa: E402

import open_clip  # noqa: E402

RATES = [
    (0.25, "index_msrvtt_fps0.25"),
    (0.5, "index_msrvtt_fps0.5"),
    (1.0, "index_msrvtt"),
    (1.5, "index_msrvtt_fps1.5"),
    (3.0, "index_msrvtt_fps3"),
]
OUT = os.path.join(ROOT, "results", "significance.csv")


def mcnemar(a, b):
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return b01, b10, 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    return b01, b10, chi2, math.erfc(math.sqrt(chi2 / 2.0))


def main():
    qdata = json.load(open(os.path.join(ROOT, "data", "msrvtt", "queries.json"),
                           encoding="utf-8"))
    # 用 1 fps 索引的片段集合确定 1000 条查询，与论文协议一致
    base = json.load(open(os.path.join(ROOT, "data", "index_msrvtt", "meta.json"),
                          encoding="utf-8"))
    clip_set = {m["clip"] for m in base}
    pairs = [(d["caption"], d["video"]) for d in qdata if d["video"] in clip_set]
    print(f"[查询] {len(pairs)} 条（与论文协议相同）")

    cfg = json.load(open(os.path.join(ROOT, "data", "index_msrvtt", "config.json"),
                         encoding="utf-8"))
    model, _, _ = open_clip.create_model_and_transforms(cfg["model"],
                                                        pretrained=cfg["pretrained"])
    model.eval()
    tok = open_clip.get_tokenizer(cfg["model"])
    texts = [p[0] for p in pairs]
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            q = model.encode_text(tok(texts[i:i + 64]))
            q = q / q.norm(dim=-1, keepdim=True)
            chunks.append(q.cpu().numpy().astype(np.float32))
    qs = np.concatenate(chunks, 0)

    hits = {}
    for rate, d in RATES:
        emb = np.load(os.path.join(ROOT, "data", d, "emb.npy"))
        meta = json.load(open(os.path.join(ROOT, "data", d, "meta.json"), encoding="utf-8"))
        h1, h10 = [], []
        for (_, vid), qi in zip(pairs, qs):
            ranked = sorted(aggregate(emb @ qi, meta, "max").items(),
                            key=lambda kv: kv[1], reverse=True)
            top = [c for c, _ in ranked[:10]]
            h1.append(1 if top[0] == vid else 0)
            h10.append(1 if vid in top else 0)
        hits[rate] = (np.array(h1), np.array(h10))
        print(f"[{rate:>4g} fps] R@1={np.mean(h1):.4f}  R@10={np.mean(h10):.4f}")

    rows = []
    print("\n[McNemar]  k  vs  rate_a  rate_b   b01  b10   p     p_Holm")
    for k, idx in (("R@1", 0), ("R@10", 1)):
        ps = []
        for i in range(len(RATES)):
            for j in range(i + 1, len(RATES)):
                ra, rb = RATES[i][0], RATES[j][0]
                b01, b10, chi2, p = mcnemar(hits[ra][idx], hits[rb][idx])
                ps.append((p, ra, rb, b01, b10, chi2))
        # Holm 校正（10 对同族检验）
        m = len(ps)
        order = sorted(range(m), key=lambda t: ps[t][0])
        adj = [0.0] * m
        running = 0.0
        for rank, t in enumerate(order):
            val = min(1.0, (m - rank) * ps[t][0])
            running = max(running, val)
            adj[t] = running
        for t, (p, ra, rb, b01, b10, chi2) in enumerate(ps):
            rows.append({"k": k, "rate_a": ra, "rate_b": rb,
                         "a_wrong_b_right": b01, "a_right_b_wrong": b10,
                         "chi2": round(chi2, 4), "p": round(p, 4),
                         "p_holm": round(adj[t], 4)})
            flag = " *" if p < 0.05 else ""
            flag += " (survives Holm)" if adj[t] < 0.05 else ""
            print(f"{k:>5} {ra:>5g} vs {rb:>5g}  {b01:>4} {b10:>4}  {p:>6.3f} "
                  f"{adj[t]:>7.3f}{flag}")
        print(f"  -> {k}: 最小未校正 p={min(x[0] for x in ps):.3f}，"
              f"最小 Holm p={min(adj):.3f}")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "rate_a", "rate_b", "a_wrong_b_right",
                                          "a_right_b_wrong", "chi2", "p", "p_holm"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[已保存] {OUT}")


if __name__ == "__main__":
    main()
