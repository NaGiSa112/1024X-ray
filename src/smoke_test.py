"""End-to-end smoke test: does the GPU, the data pipeline and training work?

    python src/smoke_test.py [--images 32]

Nothing here is meant for real training.  It

  1. runs a forward pass of a randomly initialised YOLO11n (no weight download),
  2. converts a handful of annotated train images from VOC to YOLO layout,
  3. trains for one epoch at 320 px, then runs inference on one image.

If this exits 0 the environment, the annotations and the GPU are all usable.
Everything is written under .tmp/smoke so the real project tree stays clean.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES  # noqa: E402  (single source of truth for class order)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG = ROOT / "data" / "train"
TRAIN_XML = TRAIN_IMG / "XML"
WORK = ROOT / ".tmp" / "smoke"

NAME_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def voc_to_yolo(xml_path: Path) -> tuple[str, list[str], int, int]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in NAME_TO_ID:
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{NAME_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return Path(root.findtext("filename")).stem, lines, w, h


def build_tiny_dataset(n_images: int) -> Path:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (WORK / sub).mkdir(parents=True, exist_ok=True)

    xmls = sorted(TRAIN_XML.glob("*.xml"))[:n_images]
    if not xmls:
        raise SystemExit(f"no annotations found in {TRAIN_XML}; extract train.zip first")

    kept = 0
    for i, xml_path in enumerate(xmls):
        stem, lines, _, _ = voc_to_yolo(xml_path)
        src = TRAIN_IMG / f"{stem}.jpg"
        if not src.exists() or not lines:
            continue
        split = "val" if i % 5 == 0 else "train"
        shutil.copyfile(src, WORK / "images" / split / f"{stem}.jpg")
        (WORK / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        kept += 1

    yaml_path = WORK / "data.yaml"
    # encoding="utf-8" is required: the project path contains 中文, and the
    # locale default (GBK on a Chinese Windows) would make ultralytics -- which
    # reads YAML as UTF-8 -- see a mangled path and report "images not found".
    yaml_path.write_text(
        f"path: {WORK.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8",
    )
    print(f"tiny dataset: {kept} images -> {yaml_path}")
    return yaml_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=32)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    print("=" * 62)
    print("1. forward pass (no weights, no download)")
    print("=" * 62)
    net = YOLO("yolo11n.yaml")
    net.model.to("cuda")
    t0 = time.time()
    x = torch.zeros(1, 3, 320, 320, device="cuda")
    with torch.no_grad():
        out = net.model(x)
    print(f"  output type: {type(out).__name__}, "
          f"device: {next(net.model.parameters()).device}, {time.time() - t0:.2f}s")

    print("=" * 62)
    print("2. train one epoch on a few real images")
    print("=" * 62)
    data_yaml = build_tiny_dataset(args.images)
    model = YOLO("yolo11n.yaml")
    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=320,
        batch=8,
        workers=0,          # 0 first: Windows worker processes need pipes
        device=0,
        project=str(WORK / "runs"),
        name="train",
        exist_ok=True,
        plots=False,
        val=True,
        verbose=False,
    )
    print(f"  training finished in {time.time() - t0:.1f}s")

    best = WORK / "runs" / "train" / "weights" / "best.pt"
    print(f"  checkpoint: {best} exists={best.exists()}")

    print("=" * 62)
    print("3. inference with the trained checkpoint")
    print("=" * 62)
    sample = next((WORK / "images" / "val").glob("*.jpg"))
    results = YOLO(str(best)).predict(str(sample), imgsz=320, device=0, verbose=False)
    boxes = results[0].boxes
    print(f"  {sample.name}: {0 if boxes is None else len(boxes)} boxes predicted")

    print("=" * 62)
    print("RESULT: smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
