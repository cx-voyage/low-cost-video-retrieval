"""创新点 A：按查询类型分层统计 Recall@K（scene / object / action）。

为什么要有这个实验
------------------
全局 Recall@1 = 32.2% 是一个"平均值"，它掩盖了两件事：
  1) 不同用户问的问题难度不同；
  2) 抽帧率对不同问题的伤害可能不同（动作类依赖时间，静物类不依赖）。
所以把 1000 条查询按语义类型分组，再分别统计 Recall@K。

分类规则（关键词决策表，完全可复现，没有人工逐条标注）
------------------------------------------------------
  优先级 action > scene > object：

  action —— 含"物理动作"动词（play/run/dance/swim/cook/drive/... 见 ACTION_VERBS）
  scene  —— 不含物理动作，但含场景/地点词（beach/stage/kitchen/street/... 见 SCENE_WORDS）
  object —— 其余（人物、物品、属性；含 talking-head 一类"人讲某物"的视频）

  说明：像 "a woman is talking about makeup" 这种句子，语义是"某人在讲某物"，
  画面基本静止，归到 object 而不是 action —— 因为 speech 类动词列在
  SPEECH_VERBS 里，被显式排除在动作之外。

输出
----
  results/query_types.csv    每个 (fps, 类型) 的 n / R@1 / R@5 / R@10
  results/query_ranks.json   每条查询在每个抽帧率下的名次（审计用）

用法
----
  python code/analyze_query_types.py
"""

import json
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QPATH = os.path.join(ROOT, "data", "msrvtt", "queries.json")
OUT_CSV = os.path.join(ROOT, "results", "query_types.csv")
OUT_RANKS = os.path.join(ROOT, "results", "query_ranks.json")

# (fps, 索引目录名)；注意：数据集本身是 3 fps，所以 --fps 2 实际是 1.5 fps
CONFIGS = [
    (0.25, "index_msrvtt_fps0.25"),
    (0.50, "index_msrvtt_fps0.5"),
    (1.00, "index_msrvtt"),
    (1.50, "index_msrvtt_fps1.5"),
]

# ---------------------------------------------------------------- 词表
# 物理动作动词（画面里"会动"的东西，帧率不足时最容易被抽掉）
ACTION_VERBS = """
play run walk dance jump swim cook drive ride skate surf climb fly throw kick
hit catch lift push pull carry cut pour mix stir bake fry wash clean brush
draw paint write type sew knit fold assemble build repair fix install measure
weigh pack open close turn spin wave clap fight box exercise stretch bend
squat shovel dig plant water feed groom massage juggle balance slide
shoot sprint jog hike paddle sail row dribble score serve smash pitch bat
hammer saw drill glue tape wrap peel slice chop grind blend boil grill
sing create make hold put place drop pick grab touch point hug kiss
eat drink sit stand lay lean hang extinguish light laugh smile
""".split()

# 话语类动词：出现这些词时不归为 action（画面通常静止，见文件头说明）
SPEECH_VERBS = """
talk speak say tell explain describe review show demonstrate present
interview discuss mention comment narrate introduce announce report
""".split()

# 场景/地点（刻意剔除了 road/set/field/track/court/bar/night 这类高频歧义词：
# "sky roads game"、"on the set"、"music track" 会误判成 scene）
SCENE_WORDS = """
beach street city town room kitchen bedroom bathroom office
classroom school stage studio gym pool stadium forest
mountain hill river lake sea ocean desert snow rain park garden backyard
restaurant store mall airport station highway bridge
museum library hospital church temple cafe hotel rooftop balcony factory
workshop garage farm zoo aquarium laboratory indoor outdoor sunset
underwater concert wedding party parade festival conference
""".split()


def _forms(word):
    """返回 word 的常见屈折形式（用于正则）。"""
    out = {word, word + "s", word + "es", word + "ed", word + "ing"}
    if word.endswith("e"):
        out.add(word[:-1] + "ing")
        out.add(word + "d")
    if word.endswith("y"):
        out.add(word[:-1] + "ies")
        out.add(word[:-1] + "ied")
    return out


def _pattern(words):
    alts = set()
    for w in words:
        alts |= _forms(w)
    alts = sorted(alts, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(a) for a in alts) + r")\b")


RE_ACTION = _pattern(ACTION_VERBS)
RE_SPEECH = _pattern(SPEECH_VERBS)
RE_SCENE = _pattern(SCENE_WORDS)


def classify(caption):
    """返回 'action' / 'scene' / 'object'。"""
    t = caption.lower()
    has_action = bool(RE_ACTION.search(t))
    has_speech = bool(RE_SPEECH.search(t))
    if has_action and not has_speech:
        return "action"
    if RE_SCENE.search(t):
        return "scene"
    return "object"


# ---------------------------------------------------------------- 检索
def load_index(name):
    d = os.path.join(ROOT, "data", name)
    emb = np.load(os.path.join(d, "emb.npy"))
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return emb, meta, cfg


def clip_scores(scores, meta):
    """帧分数 -> 片段分数（max 聚合，与主实验口径一致）。"""
    best = {}
    for s, m in zip(scores, meta):
        c = m["clip"]
        v = float(s)
        if c not in best or v > best[c]:
            best[c] = v
    return best


def ranks_for(emb, meta, qs, targets):
    """返回每条查询的正确片段名次（None = 前 10 名都没有）。"""
    out = []
    for q, tgt in zip(qs, targets):
        sc = clip_scores(emb @ q, meta)
        top = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)[:10]
        out.append(next((i + 1 for i, (c, _) in enumerate(top) if c == tgt), None))
    return out


def mcnemar(hits_a, hits_b):
    """同一批查询下，两种配置的配对比较（McNemar 检验，1 自由度）。

    只看"一个命中、另一个没命中"的不一致对：
      b = a 命中 & b 没命中, c = 反之
    若 b、c 差不多，说明两种配置对这批查询没有系统性差异。
    """
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return b, c, p


def main():
    queries = json.load(open(QPATH, encoding="utf-8"))
    captions = [d["caption"] for d in queries]
    targets = [d["video"] for d in queries]
    cats = [classify(c) for c in captions]

    from collections import Counter
    dist = Counter(cats)
    n_q = len(captions)
    print(f"[查询] {n_q} 条，规则分类结果：")
    for k in ("action", "scene", "object"):
        print(f"  {k:<7} {dist[k]:>4} 条  ({dist[k] / n_q:.1%})")

    print("\n[样例] 每类前 5 条（人工扫一眼就能判断规则是否合理）：")
    for k in ("action", "scene", "object"):
        print(f"  --- {k} ---")
        shown = 0
        for c, lab in zip(captions, cats):
            if lab == k and shown < 5:
                print(f"    {c[:70]}")
                shown += 1

    # ---- 文本向量：只算一次，四个索引复用 ----
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
    import torch
    import open_clip

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    _, _, cfg0 = load_index("index_msrvtt")
    model, _, _ = open_clip.create_model_and_transforms(
        cfg0["model"], pretrained=cfg0["pretrained"])
    model.eval()
    tok = open_clip.get_tokenizer(cfg0["model"])

    qs = []
    with torch.no_grad():
        for i in range(0, n_q, 64):
            t = tok(captions[i:i + 64])
            q = model.encode_text(t)
            q = q / q.norm(dim=-1, keepdim=True)
            qs.append(q.cpu().numpy().astype(np.float32))
    qs = np.concatenate(qs, 0)
    print(f"\n[编码] 查询向量 {qs.shape}")

    # ---- 逐配置检索 ----
    per_query = {}   # fps -> [rank, ...]
    rows = []
    print("\n" + "=" * 84)
    print(f"{'fps':>5} {'type':<7} {'n':>4} {'R@1':>7} {'R@5':>7} {'R@10':>7} "
          f"{'s.e.(R@1)':>10}")
    print("-" * 84)
    for fps, name in CONFIGS:
        emb, meta, cfg = load_index(name)
        ranks = ranks_for(emb, meta, qs, targets)
        per_query[fps] = ranks
        for cat in ("action", "scene", "object"):
            idx = [i for i, c in enumerate(cats) if c == cat]
            n = len(idx)
            r1 = sum(1 for i in idx if ranks[i] and ranks[i] <= 1) / n
            r5 = sum(1 for i in idx if ranks[i] and ranks[i] <= 5) / n
            r10 = sum(1 for i in idx if ranks[i] and ranks[i] <= 10) / n
            se = math.sqrt(r1 * (1 - r1) / n)
            print(f"{fps:>5g} {cat:<7} {n:>4} {r1:>6.1%} {r5:>6.1%} {r10:>6.1%} "
                  f"{se * 100:>9.1f}")
            rows.append({"fps": fps, "category": cat, "n_queries": n,
                         "recall@1": round(r1, 4), "recall@5": round(r5, 4),
                         "recall@10": round(r10, 4)})
        # 全量复核：应与 results.csv 的主结果一致
        allr1 = sum(1 for r in ranks if r and r <= 1) / n_q
        allr5 = sum(1 for r in ranks if r and r <= 5) / n_q
        allr10 = sum(1 for r in ranks if r and r <= 10) / n_q
        print(f"{'':>5} {'ALL':<7} {n_q:>4} {allr1:>6.1%} {allr5:>6.1%} {allr10:>6.1%}")
        print("-" * 84)

    # ---- 配对检验：抽帧率对"动作类"是否有可分辨的影响 ----
    print("\n[配对检验] 同一批查询、R@1，1 fps 对比其他抽帧率（McNemar，1 自由度）")
    print(f"{'类型':<7} {'对比':<14} {'b':>4} {'c':>4} {'p':>8}")
    hit = {fps: [1 if (r and r <= 1) else 0 for r in per_query[fps]]
           for fps, _ in CONFIGS}
    for cat in ("action", "scene", "object"):
        idx = [i for i, c in enumerate(cats) if c == cat]
        for fps in (0.25, 0.5, 1.5):
            a = [hit[1.0][i] for i in idx]
            b_ = [hit[fps][i] for i in idx]
            b, c, p = mcnemar(a, b_)
            print(f"{cat:<7} {f'1.0 vs {fps:g} fps':<14} {b:>4} {c:>4} {p:>8.3f}")

    # 二分对比：动作类 vs 其余（scene + object 合并，样本更大、结论更稳）
    print("\n[二分对比] action vs 其余（R@1），1 fps 对比其他抽帧率")
    print(f"{'组':<8} {'n':>4} {'对比':<14} {'b':>4} {'c':>4} {'p':>8}")
    act_idx = [i for i, c in enumerate(cats) if c == "action"]
    rest_idx = [i for i, c in enumerate(cats) if c != "action"]
    for label, idx in (("action", act_idx), ("rest", rest_idx)):
        for fps in (0.25, 0.5, 1.5):
            a = [hit[1.0][i] for i in idx]
            b_, c_, p = mcnemar(a, [hit[fps][i] for i in idx])
            print(f"{label:<8} {len(idx):>4} {f'1.0 vs {fps:g} fps':<14} "
                  f"{b_:>4} {c_:>4} {p:>8.3f}")

    # ---- 落盘 ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("fps,category,n_queries,recall@1,recall@5,recall@10\n")
        for r in rows:
            f.write(f"{r['fps']:g},{r['category']},{r['n_queries']},"
                    f"{r['recall@1']:.4f},{r['recall@5']:.4f},{r['recall@10']:.4f}\n")
    print(f"\n[已保存] {OUT_CSV}")

    audit = {"rule": "action > scene > object (keyword decision list)",
             "queries": [{"id": d.get("id"), "caption": d["caption"],
                          "category": cat, "target": tgt,
                          "ranks": {f"{fps:g}": per_query[fps][i] for fps, _ in CONFIGS}}
                         for i, (d, cat, tgt) in enumerate(zip(queries, cats, targets))]}
    with open(OUT_RANKS, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)
    print(f"[已保存] {OUT_RANKS}")


if __name__ == "__main__":
    main()
