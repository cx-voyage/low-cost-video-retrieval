"""
第 3 步：输入一句话 → 返回最匹配的视频片段

运行：
    python code/search.py "a person playing basketball"
    python code/search.py "a dog running on the beach" --topk 5

原理（三行讲完）：
    1. 用 CLIP 的文本编码器把这句话变成 512 维向量
    2. 和库里每一帧的向量算余弦相似度（向量已归一化，所以就是点积）
    3. 同一段视频的所有帧里取最高分，作为"这段视频的分数"，排序取前 k 段

为什么第 3 步要"取最高分"而不是"取平均"？
    因为一段 10 秒的视频里，可能只有 1 秒真的出现了篮球。
    取平均会被无关的帧拉低分数；取最大相当于"这段视频里只要有一帧很像，就算像"。
    这个选择本身就是一个可以写进报告的"设计决策"，第 4 步我会让你对比这两种做法。
"""

import os

# ⚠️ 必须在 import open_clip 之前设置（文本编码器也可能触发权重下载）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import argparse
import json
import time

import numpy as np
import torch

import open_clip

INDEX_DIR = os.path.join(ROOT, "data", "index")


def load_index():
    emb = np.load(os.path.join(INDEX_DIR, "emb.npy"))
    with open(os.path.join(INDEX_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(INDEX_DIR, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return emb, meta, cfg


def clip_scores(frame_scores, meta):
    """把"每帧的分数"聚合成"每段视频的分数"（取最大值）。"""
    best = {}
    for s, m in zip(frame_scores, meta):
        key = m["clip"]
        if key not in best or s > best[key][0]:
            best[key] = (float(s), m["t"], m["class"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="英文查询，用引号包起来")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    emb, meta, cfg = load_index()
    print(f"[索引] {len(meta)} 帧 / {len({m['clip'] for m in meta})} 段视频 "
          f"（模型 {cfg['model']}，抽帧 {cfg['fps']} fps）")

    model_name = cfg["model"]
    pretrained = cfg["pretrained"]
    local_ckpt = os.path.join(ROOT, "models", "clip_vitb32_laion2b.bin")
    if not os.path.isfile(pretrained) and os.path.isfile(local_ckpt):
        pretrained = local_ckpt
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    t0 = time.perf_counter()
    with torch.no_grad():
        tokens = tokenizer([args.query])
        q = model.encode_text(tokens)
        q = q / q.norm(dim=-1, keepdim=True)
        q = q.cpu().numpy().astype(np.float32)[0]
    t_encode = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    frame_scores = emb @ q           # (N,) 余弦相似度
    best = clip_scores(frame_scores, meta)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    t_search = (time.perf_counter() - t1) * 1000

    print(f"\n查询：{args.query}")
    print(f"（文本编码 {t_encode:.0f} ms，检索 {t_search:.1f} ms）\n")
    print(f"{'排名':<4}{'分数':<8}{'类别':<24}{'命中时间':<10}视频")
    print("-" * 78)
    for i, (clip, (score, t_sec, cls)) in enumerate(ranked[:args.topk], 1):
        print(f"{i:<4}{score:<8.4f}{cls:<24}{str(t_sec) + 's':<10}{clip}")


if __name__ == "__main__":
    main()
