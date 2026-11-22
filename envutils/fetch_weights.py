"""Download a file with resume + size check, HEAD-free.

    python envutils/fetch_weights.py <url> <dest> [expected_bytes]

Some hosts answer a GET instantly but never answer HEAD (huggingface.co is like
that from this machine), so the total size is taken from the `Content-Range`
header of a ranged GET instead.  Interrupted downloads resume from the current
file size.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

TIMEOUT = 60
CHUNK = 256 * 1024


def parse_total(headers: dict[str, str]) -> int | None:
    content_range = headers.get("Content-Range")
    if content_range:
        m = re.match(r"bytes \d+-\d+/(\d+)", content_range)
        if m:
            return int(m.group(1))
    length = headers.get("Content-Length")
    return int(length) if length else None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    url, dest = sys.argv[1], Path(sys.argv[2])
    expected = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dest.parent.mkdir(parents=True, exist_ok=True)

    if expected and dest.exists() and dest.stat().st_size == expected:
        print(f"already complete: {dest} ({expected} bytes)")
        return 0

    total: int | None = None
    for attempt in range(10):
        got = dest.stat().st_size if dest.exists() else 0
        if total is not None and got >= total:
            break
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={got}-"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                if total is None:
                    total = parse_total(dict(r.headers))
                    if total:
                        print(f"total {total / 1024**2:.1f} MiB -> {dest}")
                    if got:
                        print(f"  resuming from {got / 1024**2:.1f} MiB")
                mode = "ab" if got else "wb"
                t_last = time.time()
                with dest.open(mode) as fh:
                    while True:
                        block = r.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        got += len(block)
                        if time.time() - t_last > 10:
                            pct = 100.0 * got / total if total else 0
                            print(f"  {got / 1024**2:.1f}/{total / 1024**2:.1f} MiB "
                                  f"({pct:.0f}%)", flush=True)
                            t_last = time.time()
        except Exception as exc:  # noqa: BLE001 - resume on any network error
            print(f"  attempt {attempt + 1}: {type(exc).__name__}: {str(exc)[:70]}",
                  flush=True)
            time.sleep(3)

    size = dest.stat().st_size if dest.exists() else 0
    if total and size != total:
        print(f"!! incomplete: {size} != {total}")
        return 1
    if expected and size != expected:
        print(f"!! size {size} != expected {expected}")
        return 1
    print(f"done: {dest} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
