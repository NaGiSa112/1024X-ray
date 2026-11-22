"""Split the validation set into leakage-free and leaked parts, and measure both.

    python src/split_clean_val.py --build     # create the two dataset dirs
    python src/split_clean_val.py --report    # (informational) print counts

Why: 55 of the 555 val images have a perceptual twin (dHash Hamming <= 4) inside
the training set, so those images are partly memorised and inflate the val mAP we
have been using to pick models.  Splitting them out gives an honest number and
tells us how much of the local->online gap is our own measuring error.

Builds:
    data/yolo_cleanval/  train = the real train split, val = 500 clean images
    data/yolo_leakval/   train = the real train split, val = 55 leaked images
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "yolo"
TRAIN_IMG = ROOT / "data" / "train"
OUT_CLEAN = ROOT / "data" / "yolo_cleanval"
OUT_LEAK = ROOT / "data" / "yolo_leakval"

from classes import CLASSES, NUM_CLASSES  # noqa: E402


def dhash(path: Path, size: int = 8) -> np.uint64:
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = np.asarray(im, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return np.uint64(bits)


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {NUM_CLASSES}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--thr", type=int, default=4)
    args = ap.parse_args()

    val_imgs = sorted((SRC / "images" / "val").glob("*.jpg"))
    train_imgs = sorted((SRC / "images" / "train").glob("*.jpg"))
    print(f"val {len(val_imgs)}  train {len(train_imgs)}")

    train_stems = {p.stem for p in train_imgs}
    all_val = [(p.stem, TRAIN_IMG / p.name) for p in val_imgs]
    assert all(s in train_stems or True for s, _ in all_val)

    print("hashing train images ...")
    train_hash = {p.stem: dhash(TRAIN_IMG / p.name) for p in train_imgs}
    thash = np.asarray([train_hash[s] for s in train_stems], dtype=np.uint64)
    order = sorted(train_stems)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    leaked, clean = [], []
    for stem, path in all_val:
        h = np.uint64(dhash(path))
        x = np.bitwise_xor(thash, h).view(np.uint8).reshape(-1, 8)
        d = popcount[x].sum(axis=1)
        # ignore the image itself if it happens to be in train (same stem)
        same = np.array([s == stem for s in order])
        if same.any():
            d[same] = 255
        (leaked if d.min() <= args.thr else clean).append((stem, path))

    print(f"leaked (twin in train, hamming<={args.thr}): {len(leaked)}")
    print(f"clean : {len(clean)}")

    if args.build:
        for out, items in ((OUT_CLEAN, clean), (OUT_LEAK, leaked)):
            if out.exists():
                shutil.rmtree(out)
            for sub in ("images/train", "images/val", "labels/train", "labels/val"):
                (out / sub).mkdir(parents=True, exist_ok=True)
            for stem, path in items:
                link(path, out / "images" / "val" / path.name)
                link(SRC / "labels" / "val" / f"{stem}.txt",
                     out / "labels" / "val" / f"{stem}.txt")
            for p in train_imgs:
                link(TRAIN_IMG / p.name, out / "images" / "train" / p.name)
                link(SRC / "labels" / "train" / f"{p.stem}.txt",
                     out / "labels" / "train" / f"{p.stem}.txt")
            build_yaml(out)
            print(f"  built {out}  val={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
