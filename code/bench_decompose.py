"""
Micro-benchmark: split the per-frame indexing cost into video decoding and
model computation.

This exists for exactly one sentence in the report: "at least 78% of the
per-frame cost is model computation". That sentence used to be an inference
from the 27% run-to-run spread; this script measures the two components
separately so the sentence can be stated as a measurement.

Method (mirrors code/build_index.py exactly):
  * same frame sampling  : cv2.VideoCapture + CAP_PROP_POS_FRAMES seek
  * same preprocessing   : BGR->RGB, PIL, open_clip preprocess
  * same batching        : 32 frames per encode_image call, 4 torch threads
  * same split point     : decode = everything up to and including the PIL
                           image; model = preprocess + encode_image + norm

Four passes, in this order:
  1. decode only, first touch of the files   -> cold decode
  2. decode only, same files again           -> warm decode
  3. model only, on the images from pass 2   -> model
  4. full pipeline, same clips               -> end-to-end check

Run:
    . .\\env.ps1
    py code\\bench_decompose.py
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf_cache"))

import argparse
import csv
import glob
import statistics
import time

import cv2
import numpy as np
import torch
from PIL import Image

import open_clip

ROOT = _ROOT


def find_videos(data_dir, n):
    vids = sorted(glob.glob(os.path.join(data_dir, "**", "*.avi"), recursive=True))
    vids += sorted(glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True))
    return vids[:n]


def decode_clip(path, fps=1.0, max_frames=30):
    """Identical to build_index.sample_frames, but returns per-frame timings."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return [], []
    step = max(1, int(round(src_fps / max(fps, 0.01))))
    indices = list(range(0, total, step))[:max_frames]

    imgs, times = [], []
    for idx in indices:
        t0 = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        times.append((time.perf_counter() - t0) * 1000.0)
        imgs.append(im)
    cap.release()
    return imgs, times


def model_pass(model, preprocess, imgs, batch_size=32):
    """Identical to build_index.encode_images, but returns per-frame timings."""
    times = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            chunk = imgs[i:i + batch_size]
            t0 = time.perf_counter()
            batch = torch.stack([preprocess(im) for im in chunk])
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            _ = f.cpu().numpy()
            dt = (time.perf_counter() - t0) * 1000.0
            times.extend([dt / len(chunk)] * len(chunk))
    return times


def describe(name, times):
    if not times:
        return None
    return {
        "stage": name,
        "n": len(times),
        "median_ms": round(statistics.median(times), 2),
        "mean_ms": round(statistics.fmean(times), 2),
        "p90_ms": round(sorted(times)[int(len(times) * 0.9) - 1], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20, help="clips per dataset")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--dataset", type=str, default="msrvtt",
                    choices=["msrvtt", "ucf101"])
    ap.add_argument("--out", type=str, default="results/timing_components.csv")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    data_dir = (os.path.join(ROOT, "data", "msrvtt", "video")
                if args.dataset == "msrvtt"
                else os.path.join(ROOT, "data", "ucf101"))
    vids = find_videos(data_dir, args.clips)
    print(f"[data] {args.dataset}: {len(vids)} clips from {data_dir}")
    if not vids:
        raise SystemExit("[error] no videos found")

    pretrained = "laion2b_s34b_b79k"
    local_ckpt = os.path.join(ROOT, "models", "clip_vitb32_laion2b.bin")
    if os.path.isfile(local_ckpt):
        pretrained = local_ckpt
    t0 = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=pretrained)
    model.eval()
    print(f"[model] loaded in {time.perf_counter() - t0:.1f}s")

    # warm-up: one batch, so the first-touch cost is not charged to pass 1
    _ = model_pass(model, preprocess, decode_clip(vids[0], args.fps, 4)[0])

    # ---- pass 1: decode only, cold -------------------------------------
    cold, cold_imgs = [], []
    for p in vids:
        imgs, ts = decode_clip(p, args.fps, args.max_frames)
        cold.extend(ts)
        cold_imgs.append(imgs)
    print(f"[1/4] decode cold : {statistics.median(cold):.1f} ms/frame ({len(cold)} frames)")

    # ---- pass 2: decode only, warm -------------------------------------
    warm, warm_imgs = [], []
    for p in vids:
        imgs, ts = decode_clip(p, args.fps, args.max_frames)
        warm.extend(ts)
        warm_imgs.append(imgs)
    print(f"[2/4] decode warm : {statistics.median(warm):.1f} ms/frame")

    # ---- pass 3: model only, on the pass-2 images ----------------------
    mod = []
    for imgs in warm_imgs:
        mod.extend(model_pass(model, preprocess, imgs))
    print(f"[3/4] model only  : {statistics.median(mod):.1f} ms/frame")

    # ---- pass 4: full pipeline -----------------------------------------
    t0 = time.perf_counter()
    n = 0
    for p in vids:
        imgs, _ = decode_clip(p, args.fps, args.max_frames)
        _ = model_pass(model, preprocess, imgs)
        n += len(imgs)
    end_ms = (time.perf_counter() - t0) * 1000.0 / max(n, 1)
    print(f"[4/4] end-to-end  : {end_ms:.1f} ms/frame ({n} frames)")

    dec_warm = statistics.median(warm)
    mdl = statistics.median(mod)
    share = 100.0 * mdl / (dec_warm + mdl)
    print("\n" + "=" * 56)
    print(f"decode (warm) {dec_warm:6.1f} ms  |  model {mdl:6.1f} ms")
    print(f"model share of decode+model : {share:.1f}%")
    print(f"sum of parts {dec_warm + mdl:6.1f} ms  vs end-to-end {end_ms:6.1f} ms "
          f"({100.0 * (dec_warm + mdl) / end_ms:.0f}%)")
    print("=" * 56)

    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "dataset", "fps", "clips", "n_frames",
                        "decode_cold_ms", "decode_warm_ms", "model_ms",
                        "model_share_pct", "end_to_end_ms", "note"])
        w.writerow([time.strftime("%Y-%m-%d"), args.dataset, args.fps,
                    len(vids), len(mod),
                    round(statistics.median(cold), 1),
                    round(dec_warm, 1), round(mdl, 1), round(share, 1),
                    round(end_ms, 1),
                    "micro-benchmark; 4 passes; same code path as build_index.py"])
    print(f"[ok] appended -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
