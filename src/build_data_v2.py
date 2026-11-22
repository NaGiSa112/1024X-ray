"""Build a deduplicated, conflict-resolved dataset (data/yolo_v2).

    python src/build_data_v2.py                # build
    python src/build_data_v2.py --dry-run      # report only

Policies, each with a recorded reason:

  cross_split duplicate   A training copy of the same bytes as a validation image
                          leaks the validation score.  Drop the TRAINING copy, so
                          validation stays at full size.  Exception: the 3 training
                          images that also duplicate a test2 image are kept (a test
                          picture the model has seen is free score), and their
                          validation twins are removed instead.

  within-train identical  Two copies of one image with the same boxes (or boxes
  or jitter               differing by a few pixels).  Keep one copy, drop the
                          rest; either label is defensible so nothing is lost.

  within-train conflict   Two copies of one image labelled with DIFFERENT box
  (class or coordinate    sets.  The model currently receives contradictory
  mismatch)               supervision for identical pixels.  Keep the copy with
                          more boxes (usually the more complete annotation) and
                          drop the other, then list the group in the review
                          queue so a human can overrule the choice.  The image
                          itself is NOT removed, so rare classes keep their data.

Validation comes from data/yolo_clean (already leakage-free, 552 images).

Outputs
    data/yolo_v2/{images,labels}/{train,val}, data.yaml
    data/yolo_v2/build_report.json      counts, per-class boxes, every drop+reason
    data/yolo_v2/review_queue.csv       groups needing a human decision
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "yolo"
CLEAN = ROOT / "data" / "yolo_clean"
CONFLICTS = ROOT / "runs" / "audit" / "conflicts" / "summary.csv"
TRADEOFF = ROOT / "runs" / "audit" / "leak_tradeoff.json"
OUT = ROOT / "data" / "yolo_v2"

from classes import CLASSES, NUM_CLASSES  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def box_count(stem: str) -> int:
    p = SRC / "labels" / "train" / f"{stem}.txt"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--conflict-policy", choices=["keep_more", "drop_both"], default="keep_more",
                    help="what to do with duplicates whose labels genuinely differ: keep the "
                         "copy with more boxes, or drop every copy until a human decides")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    rows = list(csv.DictReader(CONFLICTS.open(encoding="utf-8-sig")))
    trade = json.loads(TRADEOFF.read_text(encoding="utf-8"))
    keep_test_overlap = set(trade["leak_train_also_in_test2"])

    drop: dict[str, str] = {}          # stem -> reason
    review: list[dict] = []
    val_extra_drop: set[str] = set()

    for r in rows:
        splits = r["splits"].split(",")
        stems = r["stems"].split(",")
        train = [s for s, sp in zip(stems, splits) if sp == "train"]
        val = [s for s, sp in zip(stems, splits) if sp == "val"]
        sev = r["severity"]

        if val:                        # cross-split duplicate
            for s in train:
                if s in keep_test_overlap:
                    val_extra_drop.update(val)   # keep train, drop the val twin
                else:
                    drop[s] = "val_leak_train_copy"
            continue

        if len(train) < 2:
            continue

        counts = [box_count(s) for s in train]
        conflicting = sev in ("class_mismatch", "coordinate_mismatch")
        best = train[counts.index(max(counts))]     # ties keep the first
        if conflicting and args.conflict_policy == "drop_both":
            # No automatic rule can tell which labelling is right, so neither is
            # used until a human looks at the side-by-side render.
            for s in train:
                drop[s] = "contradictory_label_unresolved"
        else:
            for s in train:
                if s == best:
                    continue
                drop[s] = "redundant_duplicate" if not conflicting \
                    else "contradictory_label_dropped"

        if conflicting:
            if args.conflict_policy == "drop_both":
                kept_stem, drop_stems = "", train
            else:
                kept_stem, drop_stems = best, [s for s in train if s != best]
            review.append({
                "severity": sev,
                "policy": args.conflict_policy,
                "kept": kept_stem,
                "kept_boxes": max(counts) if kept_stem else 0,
                "dropped": ";".join(drop_stems),
                "dropped_boxes": ";".join(str(c) for s, c in zip(train, counts) if s in drop_stems),
                "box_counts": r["box_counts"],
                "max_dev_px": r["max_dev_px"],
                "png": str(ROOT / "runs" / "audit" / "conflicts" / sev),
            })

    train_imgs = sorted((SRC / "images" / "train").glob("*.jpg"))
    keep = [p for p in train_imgs if p.stem not in drop]
    val_imgs = sorted((CLEAN / "images" / "val").glob("*.jpg"))
    val_imgs = [p for p in val_imgs if p.stem not in val_extra_drop]

    print(f"source train images : {len(train_imgs)}")
    print(f"dropped             : {len(drop)}")
    for reason, n in Counter(drop.values()).most_common():
        print(f"  {reason:<30}{n:>5}")
    print(f"train after         : {len(keep)}")
    print(f"val (leak-free)     : {len(val_imgs)}")

    if args.dry_run:
        print("\nDRY RUN: nothing written")
        return 0

    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for p in keep:
        link(p, out / "images" / "train" / p.name)
        shutil.copyfile(SRC / "labels" / "train" / f"{p.stem}.txt",
                        out / "labels" / "train" / f"{p.stem}.txt")
    for p in val_imgs:
        link(p, out / "images" / "val" / p.name)
        shutil.copyfile(CLEAN / "labels" / "val" / f"{p.stem}.txt",
                        out / "labels" / "val" / f"{p.stem}.txt")

    # ---- verification -------------------------------------------------
    train_hash = {sha(p) for p in (out / "images" / "train").glob("*.jpg")}
    val_hash = {sha(p): p.name for p in (out / "images" / "val").glob("*.jpg")}
    collisions = [val_hash[h] for h in train_hash if h in val_hash]
    print(f"\nverification: train contents {len(train_hash)}, val contents {len(val_hash)}, "
          f"collisions {len(collisions)}")
    if collisions:
        print(f"  !! leak remains: {collisions[:5]}")
        return 1

    boxes = Counter()
    for lbl in (out / "labels" / "train").glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                boxes[CLASSES[int(float(line.split()[0]))]] += 1
    vboxes = Counter()
    for lbl in (out / "labels" / "val").glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                vboxes[CLASSES[int(float(line.split()[0]))]] += 1

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {NUM_CLASSES}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8")

    with (out / "review_queue.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(review[0].keys()) if review else ["severity"])
        w.writeheader()
        w.writerows(review)

    report = {
        "source": str(SRC), "output": str(out),
        "policy": {
            "cross_split": "drop training copy; keep train copy only if it also matches test2",
            "identical_jitter": "keep one copy, drop the rest",
            "class_or_coordinate_conflict": "keep the copy with more boxes, flag for human review",
            "images": "hard-linked", "labels": "copied",
        },
        "counts": {
            "train_before": len(train_imgs),
            "train_after": len(keep),
            "val": len(val_imgs),
            "dropped_total": len(drop),
            "dropped_by_reason": dict(Counter(drop.values())),
            "review_groups": len(review),
            "val_twins_removed_for_test_overlap": sorted(val_extra_drop),
            "sha256_train_val_collisions": len(collisions),
        },
        "train_boxes_per_class": dict(boxes),
        "val_boxes_per_class": dict(vboxes),
        "dropped_detail": drop,
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print(f"\n{'class':<14}{'train boxes':>13}{'val boxes':>11}")
    for cls in CLASSES:
        print(f"{cls:<14}{boxes[cls]:>13}{vboxes[cls]:>11}")
    print(f"\nreview queue: {len(review)} groups -> {out / 'review_queue.csv'}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
