"""Download the torch/torchvision cu121 wheels for the DINO teacher env.

pip stalled at 0 bytes against download.pytorch.org, yet reading the same URL
with urllib sustained ~1.9 MB/s, so the download is done here and pip is only
asked to install the local files.

Downloads are resumable: a partial .part file is continued with a Range request,
so an interrupted 2.3 GB transfer does not start over.

    python src/fetch_torch_wheels.py
"""

from __future__ import annotations

import socket
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(r"E:\X光\deps\wheels")
WHL = "torch-2.1.1%2Bcu121-cp310-cp310-win_amd64.whl"
TV = "torchvision-0.16.1%2Bcu121-cp310-cp310-win_amd64.whl"
MIRRORS = [
    "https://download.pytorch.org/whl/cu121/",
    "https://mirrors.aliyun.com/pytorch-wheels/cu121/",
]

socket.setdefaulttimeout(30)


def fetch(name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name.replace("%2B", "+")
    if target.exists() and target.stat().st_size > 1024 * 1024:
        print(f"  already have {target.name} ({target.stat().st_size/1024**2:.0f} MB)")
        return target

    part = target.with_suffix(target.suffix + ".part")
    for base in MIRRORS:
        url = base + name
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "pip/26"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req) as r:
                total = int(r.headers.get("Content-Length", 0)) + have
                print(f"  {base.split('/')[2]}: {total/1024**2:.0f} MB "
                      f"(resume from {have/1024**2:.0f} MB)")
                t0 = time.time()
                got = have
                mode = "ab" if have else "wb"
                with part.open(mode) as fh:
                    last = 0.0
                    while True:
                        block = r.read(4 * 1024 * 1024)
                        if not block:
                            break
                        fh.write(block)
                        got += len(block)
                        now = time.time()
                        if now - last > 10:
                            last = now
                            rate = (got - have) / (now - t0) / 1024**2
                            print(f"    {got/1024**2:7.0f} / {total/1024**2:.0f} MB "
                                  f"({got/max(total,1)*100:5.1f}%)  {rate:.2f} MB/s", flush=True)
            part.replace(target)
            print(f"  done: {target} ({target.stat().st_size/1024**2:.0f} MB)")
            return target
        except Exception as exc:  # noqa: BLE001
            print(f"  mirror failed ({type(exc).__name__}: {str(exc)[:70]}); trying next")
    raise SystemExit("all mirrors failed")


if __name__ == "__main__":
    for name in (WHL, TV):
        print(f"==> {name}")
        fetch(name)
    print("all wheels ready in", OUT)
    sys.exit(0)
