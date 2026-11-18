"""
可选（建议挂机跑）：下载 MSR-VTT 视频包（2.19 GB）

为什么值得下：
    MSR-VTT 是文生视频检索领域最常用的标准 benchmark（1000 段测试视频 + 1000 条人工描述）。
    它的 1000 段视频内容互不重复，随机 Recall@10 只有约 1%，
    所以各项对比实验（抽帧率 / 模型 / 聚合方式）能看出真实差异。

为什么要单独写一个脚本：
    实测只有约 0.2 MB/s，2.19 GB 要下 3 小时左右，中间断线是常态。
    所以复用 dl.py 的断点续传。

用法：
    python code/download_msrvtt.py
    # 中断了就直接再跑一次，会从断点继续
"""

import os

from dl import download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")

URL = ("https://hf-mirror.com/datasets/friedrichor/MSR-VTT/"
       "resolve/main/MSRVTT_Videos.zip")
TARGET = os.path.join(RAW_DIR, "MSRVTT_Videos.zip")
EXPECTED_SIZE = 2_188_992_999      # 服务器返回的真实字节数


if __name__ == "__main__":
    print("=" * 62)
    print("MSR-VTT 视频包下载（2.19 GB，0.2 MB/s 下约需 3 小时）")
    print("随时可 Ctrl+C 中断，重跑会从断点继续")
    print("=" * 62)
    download(URL, TARGET, EXPECTED_SIZE, max_attempts=2000)
    print(f"\n保存位置：{TARGET}")
