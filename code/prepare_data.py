"""
第 1 步：下载并解压数据（UCF-101 子集）

数据来源：https://hf-mirror.com/datasets/sayakpaul/ucf101-subset
         （HuggingFace 的国内镜像，无需翻墙）
内容：若干段短视频，按"动作类别"分文件夹存放。
     文件夹名 = 类别 = 我们后面评测用的"标准答案"。

支持断点续传：断了就重跑本脚本，会从断点继续。

运行：
    python code/prepare_data.py
"""

import os
import tarfile

from dl import download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
DATA_DIR = os.path.join(ROOT, "data", "ucf101")
ARCHIVE = os.path.join(RAW_DIR, "UCF101_subset.tar.gz")

URL = ("https://hf-mirror.com/datasets/sayakpaul/ucf101-subset/"
       "resolve/main/UCF101_subset.tar.gz")
EXPECTED_SIZE = 171_386_880      # 服务器上的真实字节数


def extract(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[解压] -> {out_dir}")
    # 注意：这个文件虽然叫 .tar.gz，实际是未压缩的 tar，
    # 所以用 "r:*" 让它自己识别格式（gzip / 纯 tar 都能处理）
    with tarfile.open(path, "r:*") as tf:
        try:
            tf.extractall(out_dir, filter="data")   # Python 3.12+ 需要
        except TypeError:
            tf.extractall(out_dir)


def summarize(root):
    """递归统计每个类别有多少段视频 —— 这是后面评测的基础。"""
    counts = {}
    for cur, _, files in os.walk(root):
        vids = [f for f in files if f.lower().endswith((".avi", ".mp4", ".mkv"))]
        if not vids:
            continue
        cls = os.path.basename(cur)
        if cls in ("UCF101", "train", "val", "test"):
            cls = os.path.basename(os.path.dirname(cur))
        counts[cls] = counts.get(cls, 0) + len(vids)
    classes = sorted(counts.items())

    total = sum(n for _, n in classes)
    print(f"\n[统计] 类别数 = {len(classes)}，视频总数 = {total}")
    for name, n in classes:
        print(f"  {name:<24s} {n:>4d} 段")

    if total == 0:
        print("\n[警告] 没找到视频文件，去 data/ 下看看目录结构对不对。")


if __name__ == "__main__":
    download(URL, ARCHIVE, EXPECTED_SIZE)
    extract(ARCHIVE, DATA_DIR)

    # 有的压缩包会多套一层同名目录，这里自动往下找一层
    entries = [e for e in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, e))]
    if len(entries) == 1 and not any(
            f.lower().endswith((".avi", ".mp4"))
            for f in os.listdir(os.path.join(DATA_DIR, entries[0]))):
        DATA_DIR = os.path.join(DATA_DIR, entries[0])
        print(f"[提示] 实际数据目录：{DATA_DIR}")

    summarize(DATA_DIR)
    print("\n下一步：python code/build_index.py --max-clips 30")
