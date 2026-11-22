"""Resumable, multi-connection downloader for the PyTorch CUDA wheels.

    python envutils/fetch_wheels.py [--dest DIR] [--connections N]

This machine reaches about 1 MiB/s per connection and the upstream index is
flaky, so the 2.5 GiB torch wheel is fetched in byte ranges and can be resumed
after any interruption: each part is written to ``<wheel>.part<N>`` and a rerun
continues from the current size.  When every part is complete the parts are
concatenated into the final ``.whl`` and checked against the expected size.

Wheels already present with the right size are skipped.
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/"

WHEELS = [
    "torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl",
    "torchvision-0.26.0%2Bcu128-cp313-cp313-win_amd64.whl",
]

CHUNK = 1024 * 1024
TIMEOUT = 45


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return int(r.headers["Content-Length"])


def download_part(url: str, part: Path, start: int, end: int, label: str) -> None:
    """Fetch bytes [start, end] into part, continuing from its current size."""
    if start > end:
        return
    have = part.stat().st_size if part.exists() else 0
    if have >= end - start + 1:
        return
    begin = start + have
    req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
    attempts = 0
    while begin <= end:
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, part.open("ab") as fh:
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    begin += len(block)
            break
        except Exception:  # noqa: BLE001 - any network hiccup: resume
            attempts += 1
            if attempts > 12:
                raise
            time.sleep(min(2 * attempts, 15))
            have = part.stat().st_size if part.exists() else 0
            begin = start + have
            req = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
    print(f"  [{label}] done")


def fetch(wheel: str, dest: Path, connections: int, force: bool) -> Path:
    url = BASE + wheel
    name = urllib.parse.unquote(wheel)
    final = dest / name
    size = remote_size(url)
    print(f"{name}\n  size {size / 1024**3:.2f} GiB -> {final}")

    if final.exists() and final.stat().st_size == size and not force:
        print("  already complete, skipping")
        return final

    span = size // connections + 1
    parts = []
    for i in range(connections):
        start = i * span
        end = min(start + span - 1, size - 1)
        parts.append((dest / f"{name}.part{i}", start, end))

    threads = [
        threading.Thread(target=download_part, args=(url, p, s, e, f"part{i}"))
        for i, (p, s, e) in enumerate(parts)
    ]
    t0 = time.time()
    for t in threads:
        t.start()

    def progress() -> None:
        while any(t.is_alive() for t in threads):
            time.sleep(20)
            done = sum(p.stat().st_size for p, _, _ in parts if p.exists())
            rate = done / max(time.time() - t0, 1e-9) / 1024**2
            eta = (size - done) / max(rate, 1e-9) / 60
            print(f"  {done / 1024**3:.2f}/{size / 1024**3:.2f} GiB "
                  f"{rate:.2f} MiB/s eta {eta:.0f} min")

    watcher = threading.Thread(target=progress, daemon=True)
    watcher.start()
    for t in threads:
        t.join()

    with final.open("wb") as out:
        for p, _, _ in parts:
            with p.open("rb") as fh:
                while block := fh.read(8 * CHUNK):
                    out.write(block)
            p.unlink()

    got = final.stat().st_size
    if got != size:
        raise SystemExit(f"size mismatch for {name}: {got} != {size}")
    print(f"  complete in {(time.time() - t0) / 60:.1f} min")
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / ".tmp" / "wheels"))
    ap.add_argument("--connections", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    socket.setdefaulttimeout(TIMEOUT)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for wheel in WHEELS:
        fetch(wheel, dest, args.connections, args.force)

    print("\ninstall with:")
    print(f'  python -m pip install --no-deps "{dest}"\\*.whl')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
