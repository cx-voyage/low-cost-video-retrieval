"""
第 0 步（只做一次）：下载 CLIP 权重（605 MB）

为什么不交给 open_clip 自动下载：
    open_clip 走 huggingface_hub，遇到断网只会重试 5 次就放弃，
    而这台机器到 hf-mirror 只有约 0.2 MB/s，605 MB 要下 50 分钟，
    中途断线是常态。所以先用支持断点续传的下载器把文件拿到本地，
    之后所有脚本都直接读本地文件，不再需要联网。

用法：
    python code/get_model.py
    # 断了就再跑一次，会从断点继续
"""

import os

from dl import download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
TARGET = os.path.join(MODEL_DIR, "clip_vitb32_laion2b.bin")

URL = ("https://hf-mirror.com/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/"
       "resolve/main/open_clip_pytorch_model.bin")

# 服务器返回的真实大小（X-Linked-Size），用来校验是否下完整
EXPECTED_SIZE = 605_219_813


if __name__ == "__main__":
    print("=" * 62)
    print("CLIP ViT-B/32 (laion2b_s34b_b79k) 权重下载")
    print(f"大小 {EXPECTED_SIZE / 1e6:.0f} MB，0.2 MB/s 下约需 50 分钟")
    print("随时可以 Ctrl+C 中断，重跑会从断点继续")
    print("=" * 62)
    download(URL, TARGET, EXPECTED_SIZE)
    print(f"\n保存位置：{TARGET}")
    print("下一步：python code/prepare_data.py")
