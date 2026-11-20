"""把仓库文件同步到 GitHub（需要 token，不写进任何文件）。

用法（PowerShell）：
    $env:GITHUB_TOKEN = "github_pat_xxx"
    python code\\sync_github.py            # 只显示差异
    python code\\sync_github.py --apply    # 真正上传

只做两件事：新增/更新文件。**不删除**远程文件（避免误删）。
token 需要 fine-grained PAT：Repository access 选本仓库，Permissions → Contents: Read and write。
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

OWNER = "cx-voyage"
REPO = "low-cost-video-retrieval"
BRANCH = "main"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXCLUDE_DIRS = {"data", "models", ".venv", ".texlive", ".hf_cache", "notes",
                "__pycache__", ".git"}
EXCLUDE_EXT = {".aux", ".log", ".out", ".blg", ".toc", ".bbl", ".pyc", ".synctex"}

# 仓库里已知的、应当删除的历史文件（写死清单，只删这些，不做通配删除）
STALE = [
    "code/__pycache__/build_index.cpython-314.pyc",
    "code/__pycache__/dl.cpython-311.pyc",
    "code/__pycache__/record_result.cpython-314.pyc",
    "report/main.aux",
    "report/main.bbl",
    "report/main.blg",
    "report/main.log",
    "report/main.out",
    "report/missfont.log",
    "results/msrvtt_vitb32_fps2.json",
]


def api(path, token, method="GET", payload=None):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def local_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if parts and parts[0] in EXCLUDE_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] in EXCLUDE_EXT:
                continue
            p = os.path.join(dirpath, fn)
            out.append((os.path.relpath(p, ROOT).replace("\\", "/"), p))
    return sorted(out)


def blob_sha(data):
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正上传（默认只显示差异）")
    ap.add_argument("--prune", action="store_true", help="删除 STALE 清单里的历史文件")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("缺少环境变量 GITHUB_TOKEN")

    tree = api(f"/git/trees/{BRANCH}?recursive=1", token)
    remote = {t["path"]: t["sha"] for t in tree.get("tree", []) if t["type"] == "blob"}

    todo, same = [], 0
    for rel, p in local_files():
        data = open(p, "rb").read()
        sha = blob_sha(data)
        if remote.get(rel) == sha:
            same += 1
        else:
            todo.append((rel, data, remote.get(rel)))

    print(f"远程 {len(remote)} 个文件 | 本地一致 {same} | 需要上传 {len(todo)}")
    for rel, _, _ in todo:
        print("  ", rel)

    if not args.apply:
        print("\n（未上传。加 --apply 执行）")
        return 0

    failed = 0
    for rel, data, sha in todo:
        payload = {"message": f"update {rel}",
                   "content": base64.b64encode(data).decode(),
                   "branch": BRANCH}
        if sha:
            payload["sha"] = sha
        try:
            api(f"/contents/{urllib.parse.quote(rel)}", token, "PUT", payload)
            print("  [OK]", rel)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("  [FAIL]", rel, e)
    print(f"\n完成：{len(todo) - failed} 个成功，{failed} 个失败")

    if args.prune:
        print("\n[清理历史文件]")
        for path in STALE:
            if path not in remote:
                print("  [skip]", path, "（远程已不存在）")
                continue
            try:
                api(f"/contents/{urllib.parse.quote(path)}", token, "DELETE",
                    {"message": f"remove stale file {path}",
                     "sha": remote[path], "branch": BRANCH})
                print("  [OK]", path)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("  [FAIL]", path, e)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
