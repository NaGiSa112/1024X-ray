"""Hunt for dirty data in the X-ray dataset.

    python src/audit_dirty.py

Checks, in order of how much damage each one can do:

1. exact duplicate files (same bytes) -- within train, within test2, across
2. near-duplicate images (dHash64 Hamming distance) -- the important one: if a
   val image is a near-copy of a train image, val mAP is optimistic and the
   local->online gap is partly our own measurement error; if test2 overlaps
   train, pseudo-labelling leaks
3. unreadable / corrupt images
4. annotation geometry: degenerate boxes, boxes outside the image, extreme
   aspect ratios, boxes covering almost the whole image
5. images whose annotations all collapse to a handful of pixels

Writes runs/audit/dirty_report.json with the raw findings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
TEST = ROOT / "data" / "test2"
CSV = ROOT / "data" / "annotations.csv"
OUT = ROOT / "runs" / "audit"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path: Path, size: int = 8) -> np.uint64:
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = np.asarray(im, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return np.uint64(bits)


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two uint64 arrays, chunked over rows."""
    out = np.zeros((len(a), len(b)), dtype=np.uint8)
    for i in range(0, len(a), 256):
        x = np.bitwise_xor(a[i:i + 256, None], b[None, :])
        bytes_view = x.view(np.uint8).reshape(len(x), len(b), 8)
        out[i:i + 256] = POPCOUNT[bytes_view].sum(axis=2)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    train_imgs = sorted(TRAIN.glob("*.jpg"))
    test_imgs = sorted(TEST.glob("*.jpg"))
    print(f"train {len(train_imgs)} images, test2 {len(test_imgs)} images")

    # ---------------------------------------------------------- 1. md5
    print("\n[1/5] exact duplicate files (md5)")
    md5_map: dict[str, list[str]] = {}
    for split, imgs in (("train", train_imgs), ("test2", test_imgs)):
        for p in imgs:
            md5_map.setdefault(md5_of(p), []).append(f"{split}/{p.name}")
    dup_groups = {k: v for k, v in md5_map.items() if len(v) > 1}
    print(f"  duplicate groups: {len(dup_groups)}")
    for k, v in list(dup_groups.items())[:5]:
        print(f"    {v}")
    report["exact_duplicate_groups"] = list(dup_groups.values())

    # ------------------------------------------------------- 2. dhash
    print("\n[2/5] computing perceptual hashes")
    hashes = {}
    corrupt: list[str] = []
    for split, imgs in (("train", train_imgs), ("test2", test_imgs)):
        hs = []
        for p in imgs:
            try:
                hs.append(dhash(p))
            except Exception:  # noqa: BLE001
                corrupt.append(f"{split}/{p.name}")
                hs.append(np.uint64(0))
        hashes[split] = np.asarray(hs, dtype=np.uint64)
        print(f"  {split}: {len(hs)} hashed")
    print(f"  corrupt/unreadable: {len(corrupt)}")
    report["corrupt_images"] = corrupt

    print("\n[3/5] near-duplicate scan")
    tr = hashes["train"]
    te = hashes["test2"]
    cross = hamming_matrix_chunk(tr, te)
    for thr in (0, 2, 4, 8):
        n = int((cross <= thr).sum())
        print(f"  train<->test2 pairs with hamming <= {thr}: {n}")
    report["cross_set_near_duplicates"] = {
        str(thr): int((cross <= thr).sum()) for thr in (0, 2, 4, 8)
    }

    # worst offenders, useful to eyeball
    worst = []
    if cross.size:
        idx = np.argwhere(cross <= 4)
        for i, j in idx[:40]:
            worst.append({"train": train_imgs[int(i)].name, "test2": test_imgs[int(j)].name,
                          "hamming": int(cross[i, j])})
    report["cross_set_examples"] = worst

    within = hamming_matrix_chunk(tr, tr)
    np.fill_diagonal(within, 255)
    for thr in (0, 2, 4, 8):
        pairs = np.argwhere(within <= thr)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        print(f"  train within-set pairs with hamming <= {thr}: {len(pairs)}")
        report[f"train_within_{thr}"] = len(pairs)

    # how many val images have a near-twin in train? (leakage in the metric)
    val_dir = ROOT / "data" / "yolo" / "images" / "val"
    val_stems = {p.stem for p in val_dir.glob("*.jpg")}
    val_idx = [i for i, p in enumerate(train_imgs) if p.stem in val_stems]
    if val_idx:
        sub = within[val_idx]
        leaked = 0
        for row, vi in zip(sub, val_idx):
            others = np.delete(row, vi)
            if len(others) and others.min() <= 4:
                leaked += 1
        print(f"\n  val images with a near-twin (hamming<=4) elsewhere in train: "
              f"{leaked}/{len(val_idx)} ({leaked / len(val_idx):.1%})")
        report["val_leaked_count"] = leaked
        report["val_total"] = len(val_idx)

    # ---------------------------------------------- 4/5 annotation geometry
    print("\n[4/5] annotation geometry")
    df = pd.read_csv(CSV, dtype={"image": str})
    df["bw"] = (df.x2 - df.x1) / df.width
    df["bh"] = (df.y2 - df.y1) / df.height
    df["area"] = df.bw * df.bh
    df["ar"] = df.bw / df.bh
    issues = {
        "degenerate": int(((df.bw <= 0) | (df.bh <= 0)).sum()),
        "outside_image": int(((df.x1 < 0) | (df.y1 < 0) |
                              (df.x2 > df.width) | (df.y2 > df.height)).sum()),
        "tiny_area_lt_0.0005": int((df.area < 0.0005).sum()),
        "huge_area_gt_0.8": int((df.area > 0.8).sum()),
        "extreme_ar_gt_10_or_lt_0.1": int(((df.ar > 10) | (df.ar < 0.1)).sum()),
    }
    for k, v in issues.items():
        print(f"  {k:<28}{v}")
    report["annotation_issues"] = issues

    if issues["huge_area_gt_0.8"]:
        big = df[df.area > 0.8][["image", "class", "area"]].head(10)
        print(f"\n  boxes covering >80% of the image:\n{big.to_string(index=False)}")

    # ---------------------------------------------------- 5. per-class extremes
    print("\n[5/5] per-class extreme boxes (potential label noise)")
    print(f"  {'class':<14}{'n':>6}{'area<0.0005':>13}{'area>0.5':>10}{'ar>8':>7}")
    for cls, g in df.groupby("class"):
        print(f"  {cls:<14}{len(g):>6}{int((g.area < 0.0005).sum()):>13}"
              f"{int((g.area > 0.5).sum()):>10}{int((g.ar > 8).sum()):>7}")

    (OUT / "dirty_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {OUT / 'dirty_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
