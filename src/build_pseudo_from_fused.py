"""Build a semi-supervised training set from the ensemble's own test2 predictions.

    python src/build_pseudo_from_fused.py --fused results_test2_submit_fused.json
    python src/build_pseudo_from_fused.py --score-thr 0.5 --dry-run

Why this is the highest-value run available overnight: the test set holds 2325
images, nearly half the size of the labelled training set, and it is exactly the
distribution that gets scored.  The current 3-model ensemble reaches 75.335 online,
so its confident predictions on those images are a usable teacher signal
(transductive self-training).  The student then sees the test images themselves
instead of only the training distribution.

Precision guard: only boxes at or above --score-thr are kept (default 0.5), and an
image is included only if it keeps at least one box.  Confident ensemble boxes on
the actual scoring images are far more likely to be right than wrong, and a wrong
box would teach a false positive, so the threshold is deliberately conservative
rather than tuned for maximum pseudo-label count.

The output dataset reuses the existing leak-free 489-image validation split
untouched, so results stay comparable with every earlier number.

Conversion note: the submission file is in OFFICIAL SLOT order (knife first), while
training labels use the alphabetical training order.  The mapping goes through the
class NAME, never through a slot index.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES, NAME_TO_SLOT, SUBMISSION_ORDER  # noqa: E402

ROOT = Path(r"E:\X光")
TEST_DIR = ROOT / "data" / "test2"
BASE_TRAIN_IMG = ROOT / "data" / "yolo" / "images" / "train"
BASE_TRAIN_LBL = ROOT / "data" / "yolo" / "labels" / "train"
VAL_IMG = ROOT / "data" / "yolo_night75b" / "images" / "val"
VAL_LBL = ROOT / "data" / "yolo_night75b" / "labels" / "val"
OUT = ROOT / "data" / "yolo_semi"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default=str(ROOT / "results_test2_submit_fused.json"))
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--max-per-image", type=int, default=300)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--prefix", default="t2_",
                    help="filename prefix for pseudo-labelled test images.  MUST be non-empty: "
                         "test2 and the training set share the same 5-digit file names, so "
                         "without a prefix the pseudo labels overwrite the real GT labels of up "
                         "to 2000 training images (and the images silently fail to be added).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fused_path = Path(args.fused)
    entries = json.loads(fused_path.read_text(encoding="utf-8"))
    images = sorted(TEST_DIR.glob("*.jpg"))
    if len(entries) != len(images):
        print(f"!! fused has {len(entries)} entries but {len(images)} test images")
        return 1

    out = Path(args.out)
    # slot -> training index, resolved by class NAME
    slot_to_train = {}
    for slot, name in enumerate(SUBMISSION_ORDER):
        slot_to_train[slot] = CLASSES.index(name)

    kept_images = 0
    kept_boxes = 0
    per_class = Counter()
    decided: list[tuple[Path, list[str]]] = []

    for path, entry in zip(images, entries):
        if not entry:
            continue
        with Image.open(path) as im:
            w, h = im.size
        lines, n = [], 0
        for slot, boxes in enumerate(entry):
            tid = slot_to_train[slot]
            for box in boxes:
                if n >= args.max_per_image:
                    break
                x1, y1, x2, y2, score = box[:5]
                if score < args.score_thr:
                    continue
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                cx = max(0.0, min(1.0, (x1 + x2) / 2 / w))
                cy = max(0.0, min(1.0, (y1 + y2) / 2 / h))
                nw = max(0.0, min(1.0, bw / w))
                nh = max(0.0, min(1.0, bh / h))
                if nw <= 0 or nh <= 0:
                    continue
                lines.append(f"{tid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                per_class[CLASSES[tid]] += 1
                n += 1
        if lines:
            decided.append((path, lines))
            kept_images += 1
            kept_boxes += n

    print(f"fused entries      : {len(entries)}")
    print(f"score threshold    : {args.score_thr}")
    print(f"pseudo images kept : {kept_images} / {len(images)}")
    print(f"pseudo boxes kept  : {kept_boxes}")
    print("  per class:", dict(per_class))
    base_files = sorted(BASE_TRAIN_IMG.glob("*.jpg"))
    base = len(base_files)
    # guard: a name collision would overwrite real GT labels
    overlap = {p.name for p in base_files} & {p.name for p, _ in decided}
    if not args.prefix and overlap:
        print(f"!! {len(overlap)} test files share a name with training files and --prefix is "
              f"empty; that would overwrite {len(overlap)} real label files. Aborting.")
        return 1
    print(f"base train images  : {base}")
    print(f"name collisions    : {len(overlap)} (handled by prefix {args.prefix!r})")
    print(f"combined train     : {base + kept_images}")

    if args.dry_run:
        print("\nDRY RUN: nothing written")
        return 0
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for p in sorted(BASE_TRAIN_IMG.glob("*.jpg")):
        link(p, out / "images" / "train" / p.name)
        src = BASE_TRAIN_LBL / f"{p.stem}.txt"
        if src.exists():
            shutil.copyfile(src, out / "labels" / "train" / f"{p.stem}.txt")

    for path, lines in decided:
        pref = args.prefix
        link(path, out / "images" / "train" / f"{pref}{path.name}")
        (out / "labels" / "train" / f"{pref}{path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    for p in sorted(VAL_IMG.glob("*.jpg")):
        link(p, out / "images" / "val" / p.name)
        src = VAL_LBL / f"{p.stem}.txt"
        if src.exists():
            shutil.copyfile(src, out / "labels" / "val" / f"{p.stem}.txt")

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8")

    report = {
        "source_fused": str(fused_path),
        "score_thr": args.score_thr,
        "base_train_images": base,
        "pseudo_images": kept_images,
        "pseudo_boxes": kept_boxes,
        "combined_train_images": base + kept_images,
        "per_class_pseudo_boxes": dict(per_class),
        "val_images": len(list(VAL_IMG.glob("*.jpg"))),
        "note": "test2 images with ensemble pseudo-labels at or above score_thr",
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
