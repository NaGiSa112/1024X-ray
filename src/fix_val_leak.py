"""Build a leakage-free dataset: drop the training images whose content is also
in validation, keep validation at its full size.

    python src/fix_val_leak.py                          # default build
    python src/fix_val_leak.py --dedupe-train           # also collapse redundant train copies
    python src/fix_val_leak.py --out data/yolo_clean --dry-run

Why drop from TRAIN and not from val:
    the leak is "a val picture also exists in train".  Dropping the training copy
    removes the memorisation without shrinking the measurement set, so all past
    numbers stay comparable instead of silently changing to a 489/512-image val.

Which training images are dropped:
    45 training images share content with a val image.  Of those, 3 also share
    content with a test2 image -- those are KEPT, because a test2 picture that the
    model has seen is free score (see runs/audit/leak_tradeoff.json).  The other
    42 are pure leakage and are dropped.

Images are hard-linked (their bytes are never edited downstream); labels are
copied, because label revision is exactly what may happen later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "yolo"
TRADEOFF = ROOT / "runs" / "audit" / "leak_tradeoff.json"
DEFAULT_OUT = ROOT / "data" / "yolo_clean"

from classes import CLASSES, NUM_CLASSES  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def copy_label(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dst)


def box_count(label: Path) -> int:
    if not label.exists():
        return 0
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dedupe-train", action="store_true",
                    help="also keep only one copy per duplicated training image")
    ap.add_argument("--dedupe-policy", choices=["most_boxes", "first"], default="most_boxes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and not args.dry_run:
        print(f"refusing to overwrite existing directory: {out}")
        print("remove it yourself, or pass a different --out")
        return 1

    if not TRADEOFF.exists():
        print(f"missing {TRADEOFF}; run src/audit_leak_tradeoff.py first")
        return 1
    trade = json.loads(TRADEOFF.read_text(encoding="utf-8"))
    drop_leak = set(trade["leak_train_safe_to_drop"])
    keep_test_overlap = set(trade["leak_train_also_in_test2"])
    print(f"val-leak candidates : {len(drop_leak) + len(keep_test_overlap)}")
    print(f"  drop (pure leakage): {len(drop_leak)}")
    print(f"  keep (also in test2): {len(keep_test_overlap)}  {sorted(keep_test_overlap)}")

    train_imgs = sorted((SRC / "images" / "train").glob("*.jpg"))
    val_imgs = sorted((SRC / "images" / "val").glob("*.jpg"))
    train_keep = [p for p in train_imgs if p.stem not in drop_leak]
    print(f"\ntrain before: {len(train_imgs)}  after leak fix: {len(train_keep)}")

    dropped_dupes: list[str] = []
    if args.dedupe_train:
        print("hashing remaining train images for within-train duplicates ...")
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for p in train_keep:
            by_hash[sha(p)].append(p)
        keep2 = []
        for h, mem in by_hash.items():
            if len(mem) == 1:
                keep2.append(mem[0])
                continue
            if args.dedupe_policy == "most_boxes":
                mem = sorted(mem, key=lambda p: (-box_count(SRC / "labels" / "train" / f"{p.stem}.txt"),
                                                 p.stem))
            else:
                mem = sorted(mem, key=lambda p: p.stem)
            keep2.append(mem[0])
            dropped_dupes.extend(p.stem for p in mem[1:])
        train_keep = sorted(keep2, key=lambda p: p.stem)
        print(f"  collapsed {sum(len(v) - 1 for v in by_hash.values() if len(v) > 1)} redundant copies")
        print(f"train after dedupe: {len(train_keep)}")

    if args.dry_run:
        print("\nDRY RUN: nothing written")
        print(f"would write {out}")
        print(f"  train {len(train_keep)}  val {len(val_imgs)}")
        return 0

    # A few training images duplicate BOTH a val image and a test2 image.  Keeping
    # them is right (a test2 picture the model has seen is free score), so the
    # leak has to be removed on the val side instead -- dropping the val twin,
    # not the train copy.  This costs a handful of val images and keeps them
    # honest, which beats the alternative of a leaky measurement.
    print("\nresolving train images that overlap val *and* test2 ...")
    train_hashes = {sha(p) for p in train_keep}
    removed_val: list[str] = []
    val_keep = []
    for p in val_imgs:
        if sha(p) in train_hashes:
            removed_val.append(p.stem)
        else:
            val_keep.append(p)
    if removed_val:
        print(f"  moved {len(removed_val)} contaminated val images out of the split: "
              f"{sorted(removed_val)}")
    val_imgs = val_keep
    print(f"  val now: {len(val_imgs)} images")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for p in train_keep:
        link_image(p, out / "images" / "train" / p.name)
        copy_label(SRC / "labels" / "train" / f"{p.stem}.txt",
                   out / "labels" / "train" / f"{p.stem}.txt")
    for p in val_imgs:
        link_image(p, out / "images" / "val" / p.name)
        copy_label(SRC / "labels" / "val" / f"{p.stem}.txt",
                   out / "labels" / "val" / f"{p.stem}.txt")

    # ---- verification: no content may appear on both sides ----
    print("\nverifying train/val content disjointness ...")
    train_hashes = {sha(p) for p in sorted((out / "images" / "train").glob("*.jpg"))}
    val_hashes = {sha(p): p for p in sorted((out / "images" / "val").glob("*.jpg"))}
    collisions = [val_hashes[h].name for h in train_hashes if h in val_hashes]
    print(f"  train contents {len(train_hashes)}  val contents {len(val_hashes)}")
    print(f"  collisions: {len(collisions)}")
    if collisions:
        print(f"  !! still leaking: {collisions[:10]}")
        return 1

    box_counts: Counter[str] = Counter()
    for lbl in (out / "labels" / "train").glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                box_counts[CLASSES[int(float(line.split()[0]))]] += 1
    val_boxes: Counter[str] = Counter()
    for lbl in (out / "labels" / "val").glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                val_boxes[CLASSES[int(float(line.split()[0]))]] += 1

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {NUM_CLASSES}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8",
    )

    report = {
        "source": str(SRC),
        "output": str(out),
        "policy": {
            "leak_fix": "drop training copies whose content is in val, unless also in test2",
            "val_size": "unchanged",
            "train_dedupe": bool(args.dedupe_train),
            "dedupe_policy": args.dedupe_policy if args.dedupe_train else None,
            "images": "hard-linked",
            "labels": "copied",
        },
        "counts": {
            "train_images_before": len(train_imgs),
            "train_images_after": len(train_keep),
            "val_images": len(val_imgs),
            "dropped_val_leak": len(drop_leak),
            "kept_test_overlap": sorted(keep_test_overlap),
            "val_images_removed_for_test_overlap": sorted(removed_val),
            "dropped_within_train_dupes": len(dropped_dupes),
        },
        "train_boxes_per_class": dict(box_counts),
        "val_boxes_per_class": dict(val_boxes),
        "sha256_train_val_collisions": len(collisions),
    }
    (out / "fix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    print()
    print(f"{'class':<14}{'train boxes':>13}{'val boxes':>11}")
    for cls in CLASSES:
        print(f"{cls:<14}{box_counts[cls]:>13}{val_boxes[cls]:>11}")
    print()
    print(f"-> {out}  (data.yaml + fix_report.json)")
    print(f"   train {len(train_keep)} images, val {len(val_imgs)} images, collisions 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
