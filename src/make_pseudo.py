"""Turn a submission JSON into YOLO pseudo-labels for the test images.

    python src/make_pseudo.py --sub results_test2_submit_D_tta.json --conf 0.45

Writes:
    data/pseudo/images/*.jpg   hard-links to data/test2 (no extra disk)
    data/pseudo/labels/*.txt   YOLO labels, class ids in the TRAINING order
    data/pseudo/stats.json     how many boxes survived the confidence filter

The submission stores boxes in *submission slot* order; the labels must use the
*training* order, so the mapping goes slot -> name -> training id, never slot ->
id directly.

Low-confidence boxes are deliberately dropped: pseudo-labels are training targets,
so a wrong box actively teaches the model something false.  Only boxes the model
is reasonably sure about are kept.
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
from classes import CLASSES, NAME_TO_ID, SUBMISSION_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test2"
OUT = ROOT / "data" / "pseudo"

SLOT_TO_NAME = {i: name for i, name in enumerate(SUBMISSION_ORDER)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="submission json to harvest labels from")
    ap.add_argument("--conf", type=float, default=0.45, help="minimum score for a pseudo box")
    ap.add_argument("--max-per-image", type=int, default=100)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    sub_path = Path(args.sub)
    if not sub_path.is_absolute():
        sub_path = ROOT / sub_path
    payload = json.loads(sub_path.read_text(encoding="utf-8"))
    images = sorted(TEST_DIR.glob("*.jpg"))
    if len(payload) != len(images):
        print(f"entry count {len(payload)} != images {len(images)}")
        return 1

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    kept = Counter()
    dropped = 0
    empty_images = 0
    per_image_counts: list[int] = []

    for i, (img, entry) in enumerate(zip(images, payload)):
        w, h = Image.open(img).size
        rows = []
        for slot, boxes in enumerate(entry):
            name = SLOT_TO_NAME[slot]
            cid = NAME_TO_ID[name]
            for box in boxes:
                if len(box) < 5:
                    continue
                x1, y1, x2, y2, score = box[0], box[1], box[2], box[3], box[4]
                if score < args.conf:
                    dropped += 1
                    continue
                x1 = min(max(float(x1), 0.0), float(w))
                x2 = min(max(float(x2), 0.0), float(w))
                y1 = min(max(float(y1), 0.0), float(h))
                y2 = min(max(float(y2), 0.0), float(h))
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                if bw <= 0 or bh <= 0:
                    dropped += 1
                    continue
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                rows.append((float(score), f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"))
                kept[name] += 1

        rows.sort(key=lambda r: -r[0])
        rows = rows[: args.max_per_image]
        if not rows:
            empty_images += 1
        per_image_counts.append(len(rows))
        (OUT / "labels" / f"{img.stem}.txt").write_text(
            "\n".join(r[1] for r in rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        dst = OUT / "images" / img.name
        if not dst.exists():
            try:
                os.link(img, dst)
            except OSError:
                shutil.copyfile(img, dst)

    total = sum(kept.values())
    print(f"source        : {sub_path.name}")
    print(f"conf >= {args.conf}  ->  kept {total} boxes, dropped {dropped}")
    print(f"images with no box after filtering: {empty_images} / {len(images)}")
    if per_image_counts:
        s = sorted(per_image_counts)
        print(f"boxes/image   : min {s[0]}  median {s[len(s)//2]}  max {s[-1]}  "
              f"mean {total/len(images):.1f}")
    print()
    print(f"{'class':<14}{'pseudo boxes':>13}{'real (train)':>14}")
    real = {"drinkbottle": 1479, "glassbottle": 8736, "knife": 384, "laptop": 362,
            "metalbottle": 1134, "pressuretank": 670, "tool": 126, "umbrella": 573}
    for name in CLASSES:
        print(f"{name:<14}{kept[name]:>13}{real[name]:>14}")

    (OUT / "stats.json").write_text(json.dumps({
        "source": sub_path.name, "conf": args.conf, "total": total,
        "dropped": dropped, "empty_images": empty_images,
        "per_class": dict(kept), "per_image_counts": per_image_counts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
