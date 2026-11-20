"""
第 2 步：抽帧 + 用 CLIP 把每一帧变成向量 + 存成文件（建库）

这一步是整个系统里最"贵"的一步，也是报告里"成本"的主要来源。

运行（先激活虚拟环境，看到 (.venv) 再跑）：
    python code/build_index.py
    python code/build_index.py --fps 0.5          # 更省钱
    python code/build_index.py --max-clips 200    # 只用 200 段，跑得快

产出：
    data/index/emb.npy     每一帧的向量 (N, 512)
    data/index/meta.json   每一帧的来源（哪个视频、第几秒、什么类别）
    data/index/config.json 本次运行用了什么参数
"""

import os

# ⚠️ 必须在 import open_clip / huggingface_hub 之前设置，才能走国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf_cache"))

import argparse
import glob
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image

import open_clip

ROOT = _ROOT
DATA_DIR = os.path.join(ROOT, "data", "ucf101")
OUT_DIR = os.path.join(ROOT, "data", "index")


# ---------------------------------------------------------------- 抽帧
def sample_frames(video_path, target_fps=1.0, max_frames=30):
    """把一段视频按 target_fps 抽成若干帧。

    返回 [(时间秒数, PIL图片), ...]
    注意：视频原生可能是 25/30 帧每秒，我们只取其中 1 帧，其余全部丢弃。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []

    step = max(1, int(round(src_fps / max(target_fps, 0.01))))
    indices = list(range(0, total, step))[:max_frames]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((idx / src_fps, Image.fromarray(rgb)))

    cap.release()
    return frames


# ------------------------------------------------------------ 批量编码
def encode_images(model, preprocess, images, batch_size=32):
    """一帧一张图 → 一个 512 维向量（已归一化，方便后面直接点积算余弦相似度）。"""
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            batch = torch.stack([preprocess(im) for im in chunk])
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, 0) if feats else np.zeros((0, 512), np.float32)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0, help="每秒抽几帧（默认 1）")
    ap.add_argument("--max-clips", type=int, default=0, help="最多处理多少段视频（0=全部）")
    ap.add_argument("--max-frames", type=int, default=30, help="每段视频最多抽多少帧")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", type=str, default="",
                    help="索引输出目录（做消融实验时用不同目录，避免互相覆盖）")
    ap.add_argument("--data-dir", type=str, default="",
                    help="视频目录（默认 data/ucf101，MSR-VTT 用 data/msrvtt/video）")
    ap.add_argument("--force", action="store_true",
                    help="即使目录里已有同配置索引也强制重建（默认跳过）")
    args = ap.parse_args()

    # 4 核 8 线程的 CPU：用 4 个物理核比用 8 个逻辑线程更快（超线程对矩阵乘法帮助小）
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    # 1) 先定输出目录，再推断输入目录
    #    这一步的顺序很关键：以前 --data-dir 默认指向 UCF-101，
    #    所以 "--out-dir data\index_msrvtt" 会把 UCF 的向量写进 MSR-VTT 的索引目录，
    #    跑完就把主实验索引覆盖了。现在改为由输出目录名推断输入目录。
    out_dir = args.out_dir or OUT_DIR
    if args.data_dir:
        data_dir = args.data_dir
    elif "msrvtt" in os.path.basename(out_dir).lower():
        data_dir = os.path.join(ROOT, "data", "msrvtt", "video")
        print(f"[推断] 输出目录名含 msrvtt → 输入目录自动取 data/msrvtt/video")
    else:
        data_dir = DATA_DIR

    vids = sorted(glob.glob(os.path.join(data_dir, "**", "*.avi"), recursive=True))
    vids += sorted(glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True))
    if args.max_clips > 0:
        vids = vids[:args.max_clips]
    if not vids:
        raise SystemExit(f"[错误] 在 {data_dir} 下没找到视频。先跑 prepare_data.py")
    print(f"[数据] 共 {len(vids)} 段视频  ← {data_dir}")

    # 2) 防呆
    # (a) 同一目录已有同配置索引 → 直接跳过（避免白跑，也避免覆盖 config.json 里的建库耗时）
    # (b) 同一目录已有"不同语料"的索引 → 拒绝覆盖（避免把主实验索引写坏）
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path) and not args.force:
        try:
            old = json.load(open(cfg_path, encoding="utf-8"))
        except Exception:
            old = {}
        same = (abs(float(old.get("fps", -1)) - args.fps) < 1e-9
                and old.get("model") == args.model
                and old.get("max_frames") == args.max_frames
                and old.get("n_clips") == len(vids))
        if same:
            print(f"[跳过] {out_dir} 已有同配置索引：fps={old.get('fps')}，"
                  f"{old.get('n_frames')} 帧，建库 {old.get('build_seconds', '?')} 秒。")
            print("       索引可直接用于评测，无需重建；确实要重建请加 --force。")
            return
        raise SystemExit(
            f"[拒绝] {out_dir} 里已有索引：fps={old.get('fps')}，"
            f"{old.get('n_clips')} 段 / {old.get('n_frames')} 帧。\n"
            f"       本次要写入 {len(vids)} 段（fps={args.fps}），语料/配置不同，"
            f"直接覆盖会毁掉已有实验数据。\n"
            f"       · 想换抽帧率做消融：请用不同的 --out-dir（如 data\\index_msrvtt_fps0.5）\n"
            f"       · 确实要覆盖重建：加 --force")

    # 2) 加载模型（第一次会下载权重，约 350 MB，只下载一次）
    t0 = time.perf_counter()
    pretrained = args.pretrained
    local_ckpt = os.path.join(ROOT, "models", "clip_vitb32_laion2b.bin")
    if not os.path.isfile(pretrained) and os.path.isfile(local_ckpt):
        pretrained = local_ckpt
    print(f"[模型] 加载 {args.model} ← {pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=pretrained)
    model.eval()
    print(f"[模型] 就绪，耗时 {time.perf_counter() - t0:.1f} 秒")

    # 3) 逐段视频：抽帧 → 编码
    # 先探测源视频帧率：目标帧率高于源时，step 最小只能取 1，实际帧率被源限制。
    # 不探测就会把"目标帧率"当成"实际帧率"记进结果（本仓库曾因此在 2.0/4.0 fps 上出错）。
    src_fps_probe = 0.0
    if vids:
        _cap = cv2.VideoCapture(vids[0])
        src_fps_probe = float(_cap.get(cv2.CAP_PROP_FPS) or 0.0)
        _cap.release()
    fps_effective = min(args.fps, src_fps_probe) if src_fps_probe > 0 else args.fps
    if src_fps_probe > 0 and args.fps > src_fps_probe + 1e-6:
        print(f"[注意] 源视频只有 {src_fps_probe:.2f} fps，--fps {args.fps:g} 实际按 "
              f"{fps_effective:.2f} fps 抽帧（step=1）。记录结果时请用实际帧率！")

    all_emb, meta = [], []
    t_start = time.perf_counter()
    n_frames_total = 0

    for k, path in enumerate(vids, 1):
        cls = os.path.basename(os.path.dirname(path))
        # 目录结构可能是 <类别>/ 或 <split>/<类别>/ 或 <类别>/UCF101/
        if cls in ("UCF101", "train", "val", "test"):
            cls = os.path.basename(os.path.dirname(os.path.dirname(path)))
        frames = sample_frames(path, args.fps, args.max_frames)
        if not frames:
            print(f"  [跳过] 无法读取：{path}")
            continue

        emb = encode_images(model, preprocess, [im for _, im in frames],
                            args.batch_size)
        all_emb.append(emb)
        for (t_sec, _), i in zip(frames, range(len(frames))):
            meta.append({"clip": os.path.basename(path),
                         "class": cls,
                         "t": round(float(t_sec), 2)})
        n_frames_total += len(frames)

        if k % 10 == 0 or k == len(vids):
            elapsed = time.perf_counter() - t_start
            print(f"  [{k}/{len(vids)}] 已抽 {n_frames_total} 帧，"
                  f"用时 {elapsed:.0f}s，"
                  f"约 {elapsed / max(n_frames_total,1)*1000:.0f} ms/帧")

    emb = np.concatenate(all_emb, 0)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "emb.npy"), emb)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"fps": args.fps, "model": args.model,
                   "pretrained": pretrained,
                   "fps_effective": round(fps_effective, 3),
                   "source_fps": round(src_fps_probe, 3),
                   "max_frames": args.max_frames,
                   "n_clips": len(vids), "n_frames": len(meta)}, f,
                  ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t_start
    size_mb = emb.nbytes / 1e6

    # 把建库成本写回 config.json，供 code/record_result.py 自动汇总进 results.csv。
    # （只增加字段，不改变已有字段；旧索引没有这三项，record_result 会要求显式给 --build-seconds）
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        _cfg = json.load(f)
    _cfg.update({"build_seconds": round(elapsed, 1),
                 "ms_per_frame": round(elapsed / max(len(meta), 1) * 1000, 1),
                 "index_mb": round(size_mb, 2)})
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(_cfg, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 52)
    print(f"建库完成：{len(vids)} 段视频 / {len(meta)} 帧")
    print(f"向量矩阵：{emb.shape}，占用 {size_mb:.2f} MB")
    print(f"总耗时：{elapsed:.1f} 秒  →  {elapsed / max(len(meta),1)*1000:.0f} ms/帧")
    print(f"（这组数字请抄进 log.md，报告的成本表要用）")
    print("=" * 52)
    print("\n下一步：python code/search.py \"a person playing basketball\"")


if __name__ == "__main__":
    main()
