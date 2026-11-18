"""
准备 MSR-VTT 测试集（只解压需要的那 1000 段视频）

为什么只解压 1000 段：
    压缩包里是 10000 段视频，但标准评测只用 test 1k 划分的那 1000 段。
    解压全部要多花时间、多占 2 GB 空间，没有意义。

运行：
    python code/prepare_msrvtt.py

产出：
    data/msrvtt/video/videoXXXX.mp4   1000 段测试视频
    data/msrvtt/queries.json          1000 条 {video, caption} 查询
"""

import json
import os
import zipfile

from dl import download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
MSRVTT = os.path.join(ROOT, "data", "msrvtt")
OUT_VIDEO = os.path.join(MSRVTT, "video")

ZIP_PATH = os.path.join(RAW, "MSRVTT_Videos.zip")
JSON_PATH = os.path.join(RAW, "msrvtt_test_1k.json")
JSON_URL = ("https://hf-mirror.com/datasets/friedrichor/MSR-VTT/"
            "resolve/main/msrvtt_test_1k.json")


def main():
    # 1) 标注文件（0.3 MB）
    download(JSON_URL, JSON_PATH, None)

    entries = json.load(open(JSON_PATH, encoding="utf-8"))
    wanted = {e["video"] for e in entries}
    print(f"[标注] {len(entries)} 条查询，涉及 {len(wanted)} 段视频")

    # 2) 只解压这 1000 段
    os.makedirs(OUT_VIDEO, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = [n for n in z.namelist()
                   if os.path.basename(n) in wanted]
        print(f"[解压] 命中 {len(members)} 个文件 -> {OUT_VIDEO}")
        for i, m in enumerate(members, 1):
            target = os.path.join(OUT_VIDEO, os.path.basename(m))
            if os.path.exists(target) and os.path.getsize(target) > 0:
                continue
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            if i % 200 == 0:
                print(f"  {i}/{len(members)}")

    have = {f for f in os.listdir(OUT_VIDEO) if f.lower().endswith(".mp4")}
    print(f"[完成] 本地已有 {len(have)} 段视频")

    # 3) 只保留本地确实存在的
    queries = [e for e in entries if e["video"] in have]
    out = os.path.join(MSRVTT, "queries.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=1)
    print(f"[查询] {len(queries)} 条已保存 -> {out}")

    print("\n示例：")
    for e in queries[:3]:
        print(f"  {e['video']:<20s} {e['caption']}")

    dur = [e["end time"] - e["start time"] for e in queries
           if e.get("end time") and e.get("start time")]
    if dur:
        print(f"\n[统计] {len(dur)} 段有起止时间，平均 {sum(dur)/len(dur):.1f} 秒")

    print("\n下一步：python code/build_index.py --data-dir data/msrvtt/video "
          "--out-dir data/index_msrvtt")


if __name__ == "__main__":
    main()
