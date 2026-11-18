"""
通用下载器：支持断点续传 + 自动重试

为什么自己写：这台机器到 hf-mirror 的速度只有约 0.2 MB/s，
下几百 MB 的过程中网络中断几乎是必然的。标准库的 urlretrieve 一断就全丢，
所以必须能"从断点继续"，并且自动重试直到完成。
"""

import os
import sys
import time
import urllib.request


def human(n):
    return f"{n / 1e6:.1f} MB"


def download(url, path, expected_size=None, max_attempts=300,
             backoff=5, timeout=60, chunk=1 << 20, quiet=False):
    """下载到 path，已存在则从已有字节处续传。成功返回 True。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    name = os.path.basename(path)

    for attempt in range(1, max_attempts + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if expected_size and have >= expected_size:
            if not quiet:
                print(f"[完成] {name}（{human(have)}）")
            return True

        headers = {"User-Agent": "Mozilla/5.0"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 206 = 服务器接受续传；200 = 只能从头下
                mode = "ab" if (have > 0 and resp.status == 206) else "wb"
                if mode == "wb" and have > 0:
                    print("[续传] 服务器不支持断点，只能从头开始")
                base = 0 if mode == "wb" else have
                done = base
                t0 = time.perf_counter()
                last_print = 0.0

                with open(path, mode) as f:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        now = time.perf_counter()
                        if now - last_print > 5:
                            last_print = now
                            elapsed = max(now - t0, 1e-6)
                            speed = (done - base) / 1e6 / elapsed
                            pct = done * 100.0 / expected_size if expected_size else 0
                            eta = ((expected_size - done) / 1e6 / speed / 60
                                   if (expected_size and speed > 0) else 0)
                            sys.stdout.write(
                                f"\r  {human(done)} / {human(expected_size or 0)} "
                                f"({pct:5.1f}%)  {speed:5.2f} MB/s  "
                                f"剩余约 {eta:4.0f} 分钟   ")
                            sys.stdout.flush()

            print()
            got = os.path.getsize(path)
            if expected_size is None or got >= expected_size:
                if not quiet:
                    print(f"[完成] {name}（{human(got)}）")
                return True
            print(f"[未完成] {human(got)} / {human(expected_size)}，继续续传…")

        except Exception as e:
            print(f"\n[网络中断] {type(e).__name__}: {str(e)[:70]}")
            print(f"[重试] {backoff} 秒后从断点继续"
                  f"（第 {attempt}/{max_attempts} 次尝试）")
            time.sleep(backoff)

    raise RuntimeError(f"多次重试仍失败：{path}")
