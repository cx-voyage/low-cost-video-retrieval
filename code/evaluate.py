"""
第 4 步：算 Recall@K —— 这是报告里最重要的一个数字

支持两种评测协议：

1) --protocol ucf（默认）
   UCF-101 子集，视频按动作类别分文件夹，类别名就是标准答案。
   查询例： "a video of playing basketball" -> 正确 = 所有 Basketball 视频

2) --protocol msrvtt
   MSR-VTT test 1k，1000 段互不重复的视频 + 1000 条人工描述。
   查询例： "a woman creating a fondant baby and flower" -> 正确 = video7020.mp4
   每段视频只有 1 条正确答案，所以随机 Recall@10 只有 1%，
   各项设计选择的差异才测得出来。

指标定义：
   Recall@K = 前 K 个结果里"至少包含 1 个正确视频"的查询所占比例

运行：
    python code/evaluate.py --topk 10
    python code/evaluate.py --index-dir data/index_msrvtt --protocol msrvtt \
        --queries data/msrvtt/queries.json --save results/msrvtt_vitb32_1fps.json
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import argparse
import json
import re

import numpy as np
import torch

import open_clip

INDEX_DIR = os.path.join(ROOT, "data", "index")


def camel_to_words(name):
    """'ApplyEyeMakeup' -> 'apply eye makeup'（UCF 类别名是驼峰写法）"""
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return s.replace("_", " ").lower().strip()


def make_query(cls):
    return f"a video of {camel_to_words(cls)}"


def aggregate(frame_scores, meta, method="max"):
    """每帧分数 -> 每段视频分数。method = 'max' 或 'mean'"""
    buckets = {}
    for s, m in zip(frame_scores, meta):
        buckets.setdefault(m["clip"], []).append(float(s))
    if method == "max":
        return {k: max(v) for k, v in buckets.items()}
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def random_recall(n_items, n_relevant, k):
    """随机排序时 Recall@K 的理论期望（基线）。"""
    if n_relevant <= 0:
        return 0.0
    p_none = 1.0
    for i in range(min(k, n_items)):
        p_none *= (n_items - n_relevant - i) / (n_items - i)
    return 1.0 - p_none


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--index-dir", type=str, default="")
    ap.add_argument("--protocol", type=str, default="ucf", choices=["ucf", "msrvtt"])
    ap.add_argument("--queries", type=str, default="")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    index_dir = args.index_dir or INDEX_DIR
    emb = np.load(os.path.join(index_dir, "emb.npy"))
    with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(index_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    clips = sorted({m["clip"] for m in meta})
    clip_set = set(clips)
    clip_class = {m["clip"]: m["class"] for m in meta}
    n_items = len(clips)

    # ---- 构造 (查询文本, 正确答案集合) 列表 ----
    if args.protocol == "msrvtt":
        qpath = args.queries or os.path.join(ROOT, "data", "msrvtt", "queries.json")
        qdata = json.load(open(qpath, encoding="utf-8"))
        pairs = [(d["caption"], {d["video"]}) for d in qdata
                 if d["video"] in clip_set]
        print(f"[协议] MSR-VTT test 1k（每查询 1 个正确答案）")
    else:
        classes = sorted({m["class"] for m in meta})
        pairs = [(make_query(c),
                  {c2 for c2 in clips if clip_class[c2] == c})
                 for c in classes]
        print(f"[协议] UCF-101 子集（按类别，每查询多个正确答案）")

    print(f"[索引] {len(meta)} 帧 / {n_items} 段视频")
    print(f"[配置] 模型 {cfg['model']}，抽帧 {cfg['fps']} fps，聚合 {args.agg}")
    print(f"[查询] {len(pairs)} 条\n")

    model, _, _ = open_clip.create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"])
    model.eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"])

    texts = [p[0] for p in pairs]
    qs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            t = tokenizer(texts[i:i + 64])
            q = model.encode_text(t)
            q = q / q.norm(dim=-1, keepdim=True)
            qs.append(q.cpu().numpy().astype(np.float32))
    qs = np.concatenate(qs, 0)

    K = args.topk
    ks = [k for k in (1, 5, 10) if k <= K]
    hits = {k: 0 for k in ks}
    rnd = {k: 0.0 for k in ks}
    details = []

    for (text, relevant), qi in zip(pairs, qs):
        scores = emb @ qi
        clip_score = aggregate(scores, meta, args.agg)
        ranked = sorted(clip_score.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in ranked[:K]]

        for k in ks:
            ok = any(c in relevant for c in top[:k])
            hits[k] += int(ok)
            rnd[k] += random_recall(n_items, len(relevant), k)

        rank = next((i + 1 for i, c in enumerate(top) if c in relevant), None)
        details.append({"query": text, "rank": rank, "top1": top[0] if top else None})

    n_q = len(pairs)
    print(f"{'指标':<12}{'本系统':<12}{'随机基线':<12}{'提升'}")
    print("-" * 48)
    result = {"protocol": args.protocol, "agg": args.agg,
              "n_clips": n_items, "n_queries": n_q, "config": cfg}
    for k in ks:
        r = hits[k] / n_q
        b = rnd[k] / n_q
        result[f"Recall@{k}"] = round(r, 4)
        result[f"random@{k}"] = round(b, 4)
        print(f"Recall@{k:<5}{r:>8.1%}    {b:>8.1%}    "
              f"{(r / b if b > 0 else float('inf')):>6.1f}x")

    if args.protocol == "msrvtt":
        r1 = result.get("Recall@1", 0)
        print(f"\n[诊断] 1000 条查询里，正确答案排第 1 的有 {round(r1 * n_q)} 条")
        print("\n命中示例：")
        for d in [d for d in details if d["rank"] and d["rank"] <= 3][:5]:
            print(f"  第{d['rank']}名  {d['query'][:66]}")
        print("\n失败示例（正确答案落在前 K 名之外）：")
        for d in [d for d in details if not d["rank"]][:5]:
            print(f"  未命中    {d['query'][:60]}  (top1={d['top1']})")
    else:
        print("\n每个类别的首个命中位置（None = 前 K 名里一个都没中）：")
        for d in details:
            fh = d["rank"]
            flag = "" if fh else "   <- 未命中"
            print(f"  {d['query'][11:]:<24s} 首个命中 {str(fh):<6s}{flag}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[已保存] {args.save}")

    print("\n（把上面的 Recall 抄进 log.md 和 results/results.csv）")


if __name__ == "__main__":
    main()
