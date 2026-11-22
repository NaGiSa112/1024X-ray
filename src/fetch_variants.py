"""Download the alternative domain-pretrained YOLO variants and check they load.

    python src/fetch_variants.py

The DINO checkpoint from the same paper turned out to score only 0.368 AP50 on our
validation set versus 0.786 for our own model, so "borrow a strong teacher" needs
weights that (a) share our task's domain pretraining and (b) are a different
architecture from the YOLOv8l we already have.  These five variants are released
in ultralytics format with EDS-1 pretraining, so they are drop-in instead of
requiring a second framework.

Loadability is checked, not assumed: custom backbones (NextViT, HGNetV2, DOAM)
only work if this ultralytics version knows their modules.
"""

from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path

ROOT = Path(r"E:\X光")
OUT = ROOT / "weights"
BASE = "https://huggingface.co/jgen/xray-comparative-evaluation/resolve/main/"

VARIANTS = {
    "nextvit_eds1": "nextvit_yolov8/nextvit_yolov8_EDS-1/best.pt",
    "hr_eds1": "yolov8_HR/yolov8_HR_EDS-1/best.pt",
    "hgnetv2_eds1": "yolov8_HGNetV2/yolov8_HGNetV2_EDS-1/best.pt",
    "lim_eds1": "yolov8_LIM/yolov8_LIM_EDS-1/best.pt",
    "doam_eds1": "yolov8_DOAM/yolov8_DOAM_EDS-1/best.pt",
}

socket.setdefaulttimeout(60)


def download(name: str, remote: str) -> Path | None:
    dst = OUT / f"{name}.pt"
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        print(f"  have {dst.name} ({dst.stat().st_size/1024**2:.0f} MiB)")
        return dst
    url = BASE + remote
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dsh"})
        with urllib.request.urlopen(req) as r, dst.open("wb") as fh:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            while True:
                b = r.read(8 * 1024 * 1024)
                if not b:
                    break
                fh.write(b)
                got += len(b)
        print(f"  {dst.name}: {got/1024**2:.0f} MiB (expected {total/1024**2:.0f})")
        return dst
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}: download failed {type(exc).__name__} {str(exc)[:60]}")
        dst.unlink(missing_ok=True)
        return None


def main() -> int:
    from ultralytics import YOLO

    results = {}
    for name, remote in VARIANTS.items():
        print(f"==> {name}")
        path = download(name, remote)
        if not path:
            results[name] = {"loadable": False, "reason": "download failed"}
            continue
        try:
            model = YOLO(str(path))
            names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
            params = sum(p.numel() for p in model.model.parameters()) / 1e6
            info = {"loadable": True, "params_M": round(params, 2),
                    "nc": len(names), "names": names[:12],
                    "task": model.task, "size_MiB": round(path.stat().st_size / 1024**2, 1)}
            print(f"    LOADS  {params:.2f}M params, {len(names)} classes")
            print(f"    names: {names}")
        except Exception as exc:  # noqa: BLE001
            info = {"loadable": False, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
            print(f"    FAILED {info['reason'][:120]}")
        results[name] = info

    (ROOT / "runs" / "audit" / "variant_load_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [k for k, v in results.items() if v.get("loadable")]
    print(f"\nloadable: {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
