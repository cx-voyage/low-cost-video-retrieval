"""数字终审：把 main.tex 里各表格的每个数字，与 results/*.csv 逐格对照。

为什么要有这个脚本
------------------
报告里最容易出错的地方不是正文文字，而是**表格里的数字**：改了一次实验、忘了同步表，
或者手抄时少一位。人眼核对 20 多个数字很容易漏，脚本不会漏。

用法：
    python code/audit_numbers.py

输出：每张表逐格 PASS / DIFF；只要有一处 DIFF 就以非零码退出，方便当作发布前闸门。
"""

import csv
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "report", "main.tex")
CSV_MAIN = os.path.join(ROOT, "results", "results.csv")
CSV_QTYPE = os.path.join(ROOT, "results", "query_types.csv")

TOL = 0.06          # 显示精度 0.1 时的容差


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tex_table(tex, label):
    """取出 \\label{label} 所在 table 环境的正文行（含 & 的行）。"""
    i = tex.find(r"\label{%s}" % label)
    if i < 0:
        return []
    start = tex.rfind(r"\begin{table}", 0, i)
    end = tex.find(r"\end{table}", i)
    body = tex[start:end]
    rows = []
    for line in body.splitlines():
        if "&" in line:
            rows.append(line.rstrip("\\ ").split("&"))
    return rows


def num(cell):
    """把 LaTeX 单元格转成浮点；不是纯数字则返回 None。"""
    s = cell.strip().replace(r"\,", "")
    s = s.replace("\\", "").replace("{,}", "").replace("$", "").replace("%", "")
    s = s.replace("--", "-").strip()
    if s.endswith("pp"):
        s = s[:-2].strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    return float(s)


def check(name, got, want, tol=TOL):
    global FAILED
    if got is None or want is None:
        return
    ok = abs(got - want) <= tol
    if not ok:
        FAILED += 1
    print(f"  {'PASS' if ok else 'DIFF'}  {name:<38} tex={got:<10g} csv={want:<10g}")


FAILED = 0


def main():
    tex = open(TEX, encoding="utf-8").read()
    main_rows = load_csv(CSV_MAIN)
    ms = {(float(r["fps"]), r["agg"]): r for r in main_rows
          if r["dataset"] == "msrvtt_1k" and r["build_seconds"]}
    qtype = {(float(r["fps"]), r["category"]): r for r in load_csv(CSV_QTYPE)}

    # ---------------- Table III: sampling-rate ablation ----------------
    print("\n[Table tab:fps] 抽帧率消融")
    for cells in tex_table(tex, "tab:fps"):
        fps = num(cells[0])
        if fps is None or (fps, "max") not in ms:
            continue
        r = ms[(fps, "max")]
        vals = [num(c) for c in cells[1:]]
        nums = [v for v in vals if v is not None]
        want = [float(r["n_frames"]), float(r["build_seconds"]), float(r["index_mb"]),
                float(r["recall@1"]) * 100, float(r["recall@5"]) * 100,
                float(r["recall@10"]) * 100]
        labels = ["frames", "build_s", "index_mb", "R@1", "R@5", "R@10"]
        for lab, g, w in zip(labels, nums, want):
            check(f"{fps:g} fps {lab}", g, w)

    # ---------------- Table II: main results ----------------
    print("\n[Table tab:main] 主结果（1 fps）")
    r = ms[(1.0, "max")]
    for cells in tex_table(tex, "tab:main"):
        metric = cells[0].strip().lower()
        k = {"recall@1": "1", "recall@5": "5", "recall@10": "10"}.get(metric)
        if not k:
            continue
        vals = [num(c) for c in cells[1:]]
        nums = [v for v in vals if v is not None]
        if len(nums) >= 3:
            check(f"Recall@{k} this work", nums[0], float(r[f"recall@{k}"]) * 100)
            check(f"Recall@{k} random", nums[1], float(r[f"rand@{k}"]) * 100)
            gap = (float(r[f"recall@{k}"]) - float(r[f"rand@{k}"])) * 100
            check(f"Recall@{k} gap", nums[2], gap)

    # ---------------- Table V: query types ----------------
    print("\n[Table tab:qtype] 查询类型分层")
    for cells in tex_table(tex, "tab:qtype"):
        cat = cells[0].strip()
        if cat not in ("action", "scene", "object", "all"):
            continue
        vals = [num(c) for c in cells[1:]]
        nums = [v for v in vals if v is not None]
        if cat == "all":
            n = 1000
            for fps_i, fps in enumerate((0.25, 0.5, 1.0, 1.5)):
                if fps not in [k[0] for k in ms]:
                    continue
                rr = ms.get((fps, "max"))
                if not rr:
                    continue
                check(f"all {fps:g} fps R@1", nums[1 + fps_i],
                      float(rr["recall@1"]) * 100)
            check("all 1 fps R@10", nums[-1], float(ms[(1.0, "max")]["recall@10"]) * 100)
        else:
            n = nums[0]
            for fps_i, fps in enumerate((0.25, 0.5, 1.0, 1.5)):
                q = qtype.get((fps, cat))
                if not q:
                    continue
                if fps_i == 0:
                    check(f"{cat} n", n, float(q["n_queries"]))
                check(f"{cat} {fps:g} fps R@1", nums[1 + fps_i],
                      float(q["recall@1"]) * 100)
            check(f"{cat} 1 fps R@10", nums[-1], float(qtype[(1.0, cat)]["recall@10"]) * 100)

    # ---------------- 正文里的派生数字（最容易变质） ----------------
    print("\n[正文派生数字] Abstract / §V-C 的倍数与差值")
    flat = re.sub(r"\s+", " ", tex)
    rows = sorted(ms.values(), key=lambda x: float(x["build_seconds"]))
    idx = {float(x["fps"]): x for x in ms.values()}
    lo, hi = rows[0], rows[-1]
    cost_span = float(hi["build_seconds"]) / float(lo["build_seconds"])
    rate_span = float(hi["fps"]) / float(lo["fps"])
    r1 = {f: float(x["recall@1"]) * 100 for f, x in idx.items()}
    r1_span = max(r1.values()) - min(r1.values())

    def claim(label, pattern, want, tol=0.06):
        m = re.search(pattern, flat)
        if not m:
            print(f"  SKIP  {label:<38} 正文没匹配到该句式")
            return
        raw = m.group(1).replace("{,}", "").replace(",", "")
        check(label, float(raw), want, tol=tol)

    claim("abstract rate span", r"(?:across|over) a ([0-9.]+)\$\\times\$ range of sampling rates", rate_span)
    claim("abstract cost span", r"changes indexing cost by ([0-9.]+)\$\\times\$", cost_span, tol=0.5)
    claim("contrib cost span", r"indexing cost changes by ([0-9.]+)\$\\times\$", cost_span, tol=0.5)
    claim("abstract R@1 span", r"Recall@1 moves by ([0-9.]+)\\,pp", r1_span)
    claim("1 fps vs 0.25 fps cost", r"indexing cost fell by ([0-9.]+)\$\\times\$",
          float(idx[1.0]["build_seconds"]) / float(idx[0.25]["build_seconds"]))
    claim("1.5 fps vs 1 fps cost", r"cost grows ([0-9.]+)\$\\times\$ \(1\{,\}759",
          float(idx[1.5]["build_seconds"]) / float(idx[1.0]["build_seconds"]))
    claim("1.5 fps index growth", r"and the index ([0-9.]+)\\%",
          (float(idx[1.5]["index_mb"]) / float(idx[1.0]["index_mb"]) - 1) * 100, tol=0.55)
    claim("3 fps vs 1 fps cost", r"grows ([0-9.]+)\$\\times\$ \(3\{,\}927",
          float(idx[3.0]["build_seconds"]) / float(idx[1.0]["build_seconds"]))
    claim("3 fps frame ratio", r"frame count grows (?:by )?([0-9.]+)\$\\times\$ rather than",
          float(idx[3.0]["n_frames"]) / float(idx[1.0]["n_frames"]))

    # 贡献列表与 §VI-A 的两处对比数字
    act = float(qtype[(1.0, "action")]["recall@1"]) * 100
    obj = float(qtype[(1.0, "object")]["recall@1"]) * 100
    claim("VI-A action-object gap", r"a gap of ([0-9.]+)\\,pp \(\$z=2\.8\$", act - obj)
    claim("VI-A pooled range", r"more than four times the ([0-9.]+)\\,pp that the pooled metric",
          r1_span)

    r10 = {f: float(x["recall@10"]) * 100 for f, x in idx.items()}
    vals = list(r10.values())
    max_r10 = max(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:])
    claim("largest R@10 change", r"largest uncorrected change is ([0-9.]+)\\,pp of Recall@10",
          max_r10)

    # 每 pp 成本的上下界（§VI-B 与贡献列表；只看 1 fps 以下的步长）
    pairs = sorted(idx.values(), key=lambda x: float(x["fps"]))
    costs = []
    for a, b in zip(pairs, pairs[1:]):
        if float(b["fps"]) > 1.0 + 1e-9:
            continue
        dsec = float(b["build_seconds"]) - float(a["build_seconds"])
        dpp = (float(b["recall@1"]) - float(a["recall@1"])) * 100
        if dpp > 1e-9:
            costs.append(dsec / dpp)
    print(f"  INFO  每 pp 成本 {min(costs):.0f}--{max(costs):.0f} s")
    m = re.search(r"between ([0-9]+) and ([0-9]+)\\,s of", flat)
    if m:
        check("cost/point low", float(m.group(1)), min(costs), tol=1.5)
        check("cost/point high", float(m.group(2)), max(costs), tol=1.5)
    else:
        print("  SKIP  cost per point bounds")
    claim("1.5->3 fps extra seconds", r"costs a further ([0-9\{\},]+)\\,s and returns ([0-9.]+)",
          float(idx[3.0]["build_seconds"]) - float(idx[1.5]["build_seconds"]), tol=0.55)

    # ---------------- 正文里的 p 值（significance.csv） ----------------
    print("\n[配对检验 p 值] 正文 vs results/significance.csv")
    sig_path = os.path.join(ROOT, "results", "significance.csv")
    if not os.path.isfile(sig_path):
        print("  SKIP  找不到 significance.csv（先跑 code/analyze_significance.py）")
    else:
        sig = {}
        with open(sig_path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                a, b = sorted((float(r["rate_a"]), float(r["rate_b"])))
                sig[(r["k"], a, b)] = r

        def p_of(k, a, b):
            return float(sig[(k, a, b)]["p"])

        min_holm = min(float(v["p_holm"]) for v in sig.values())
        min_raw = min(float(v["p"]) for v in sig.values())
        claim("p: 1.0 vs 0.25 R@1", r"paired test gives \$p=([0-9.]+)\$ for Recall@1",
              p_of("R@1", 0.25, 1.0), tol=0.006)
        claim("p: endpoints 0.25 vs 3.0 R@1",
              r"cannot separate (?:that difference )?from\s+zero \(\$p=([0-9.]+)\$\)",
              p_of("R@1", 0.25, 3.0), tol=0.006)
        claim("smallest Holm-corrected p", r"smallest corrected \$p=([0-9.]+)\$", min_holm, tol=0.006)
        claim("p: 1.0 vs 3.0 R@10", r"65\.2\\% \(\$p=([0-9.]+)\$\)",
              p_of("R@10", 1.0, 3.0), tol=0.006)

        # 配对差值的 95% CI（Wald，只用不一致对）
        row = sig[("R@1", 1.0, 3.0)]
        b01 = int(row["a_wrong_b_right"])
        b10 = int(row["a_right_b_wrong"])
        n_q = 1000
        diff = (b01 - b10) / n_q * 100
        nd = b01 + b10
        se = math.sqrt(nd - (b01 - b10) ** 2 / n_q) / n_q * 100
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        m = re.search(r"confidence\s+interval of \$\[([-+0-9.]+),([-+0-9.]+)\]\$", flat)
        if not m:
            print("  SKIP  CI (1.0 vs 3.0 R@1)                正文没匹配到")
        else:
            check("CI lower (1.0 vs 3.0 R@1)", float(m.group(1)), lo, tol=0.15)
            check("CI upper (1.0 vs 3.0 R@1)", float(m.group(2)), hi, tol=0.15)
            print(f"  INFO  diff={diff:+.1f} pp, 95%CI [{lo:+.1f},{hi:+.1f}]")

        # 每条“答对”的成本
        per_correct = {f: float(idx[f]["build_seconds"]) / (float(idx[f]["recall@1"]) * 1000)
                       for f in (0.25, 1.0, 3.0)}
        m = re.search(r"costs ([0-9.]+)\\,s per correct retrieval at 0\.25\\,fps,\s+"
                      r"([0-9.]+)\\,s at 1\\,fps and ([0-9.]+)\\,s at 3\\,fps", flat)
        if not m:
            print("  SKIP  cost per correct retrieval        正文没匹配到")
        else:
            for got, f in zip(m.groups(), (0.25, 1.0, 3.0)):
                check(f"cost/correct {f:g} fps", float(got), per_correct[f], tol=0.02)

        # 排名分档与 caption 长度的实测值
        qr = json.load(open(os.path.join(ROOT, "results", "query_ranks.json"),
                            encoding="utf-8"))["queries"]
        rk = [x["ranks"]["1"] for x in qr]
        b25 = 100 * sum(1 for r in rk if r is not None and 2 <= r <= 5) / len(rk)
        b610 = 100 * sum(1 for r in rk if r is not None and 6 <= r <= 10) / len(rk)
        m = re.search(r"ranks 2 and 10[^0-9]*([0-9.]+)\\% of\s+them in ranks 2--5 and "
                      r"([0-9.]+)\\% in 6--10", flat)
        if m:
            check("rank bucket 2-5", float(m.group(1)), b25, tol=0.06)
            check("rank bucket 6-10", float(m.group(2)), b610, tol=0.06)
        else:
            print("  SKIP  rank buckets                     正文没匹配到")

        def r1_of(words):
            v = [1 if x["ranks"]["1"] == 1 else 0 for x in qr
                 if len(x["caption"].split()) in words]
            return 100 * sum(v) / len(v)

        m = re.search(r"is ([0-9.]+)\\% for six to eight\s+words and ([0-9.]+)\\% for "
                      r"twelve to fifteen", flat)
        if m:
            check("caption 6-8 words R@1", float(m.group(1)), r1_of(range(6, 9)), tol=0.06)
            check("caption 12-15 words R@1", float(m.group(2)), r1_of(range(12, 16)), tol=0.06)
        else:
            print("  SKIP  caption-length buckets           正文没匹配到")

        print(f"  INFO  最小未校正 p={min_raw:.3f}，最小 Holm p={min_holm:.3f}"
              f"（{len(sig)} 对检验）")

    print("\n" + ("=" * 60))
    if FAILED:
        print(f"共有 {FAILED} 处不一致 —— 发布前必须修好")
        sys.exit(1)
    print("全部一致（0 处 DIFF）")


if __name__ == "__main__":
    main()
