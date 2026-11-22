"""Export predictions for data/test2 in the official submission format.

    python src/export_submission.py --weights runs/first_test/A_coco640/weights/best.pt

Writes `results_test2_submit.json`: a list of 2325 entries, one per test image in
file-name order, each entry being 8 lists (one per submission slot) of boxes
`[x1, y1, x2, y2, score]` in original image pixels.

The mapping from the model's own class ids to the submission slots goes through
the class *name* (`src/classes.py::NAME_TO_SLOT`), never through raw ids: the
training ids are alphabetical while the official slots are not (knife is training
id 2 but slot 0, glassbottle is training id 1 but slot 2).

Use --dry-run to check the plumbing on a handful of images without writing a
full submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import NAME_TO_SLOT, SUBMISSION_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test2"
TEMPLATE = ROOT / "results_test2.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default=str(TEST_DIR))
    ap.add_argument("--out", default=str(ROOT / "results_test2_submit.json"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="keep low so the scorer can pick its own threshold")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--augment", action="store_true",
                    help="test-time augmentation (multi-scale + flip); slower, usually a small gain")
    ap.add_argument("--square", action="store_true",
                    help="force square padding, including same-shape batches, for comparison with --square validation")
    ap.add_argument("--device", default="0")
    ap.add_argument("--dry-run", type=int, default=0,
                    help="only process this many images, write to <out>.dryrun.json")
    args = ap.parse_args()

    source = Path(args.source)
    images = sorted(source.glob("*.jpg"))
    if not images:
        print(f"no images in {source}")
        return 1

    expected = None
    if TEMPLATE.exists():
        expected = len(json.loads(TEMPLATE.read_text(encoding="utf-8")))
        if expected != len(images):
            print(f"!! template has {expected} entries but {len(images)} test images")
            return 1

    if args.dry_run:
        images = images[: args.dry_run]

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names: dict[int, str] = model.names
    unknown = [n for n in names.values() if n not in NAME_TO_SLOT]
    if unknown:
        print(f"!! model classes not in the submission order: {unknown}")
        print(f"   submission order is: {SUBMISSION_ORDER}")
        return 1

    print(f"model classes : {names}")
    print(f"images        : {len(images)}")
    print(f"slots         : {SUBMISSION_ORDER}")

    entries: list[list[list[list[float]]]] = []
    total_boxes = 0
    skipped = 0

    for i in range(0, len(images), args.batch):
        chunk = images[i: i + args.batch]
        results = model.predict(
            [str(p) for p in chunk],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            augment=args.augment,
            rect=not args.square,
            verbose=False,
        )
        for res in results:
            entry: list[list[list[float]]] = [[] for _ in range(len(SUBMISSION_ORDER))]
            boxes = res.boxes
            h, w = res.orig_shape[:2]
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, classes):
                    # The official rule is "coordinates must be > 0 and must not
                    # exceed the image width/height", so clamp into (0, w] rather
                    # than [0, w] -- a box touching x=0 would violate "> 0".
                    eps = 1e-3
                    x1 = round(min(max(float(x1), eps), float(w)), 3)
                    y1 = round(min(max(float(y1), eps), float(h)), 3)
                    x2 = round(min(max(float(x2), eps), float(w)), 3)
                    y2 = round(min(max(float(y2), eps), float(h)), 3)
                    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
                        skipped += 1
                        continue
                    slot = NAME_TO_SLOT[names[int(cls)]]
                    entry[slot].append([x1, y1, x2, y2, round(float(score), 6)])
                    total_boxes += 1
            entries.append(entry)
        done = min(i + args.batch, len(images))
        print(f"  {done}/{len(images)} images, {total_boxes} boxes")

    out = Path(args.out)
    if args.dry_run:
        out = out.with_suffix(".dryrun.json")
    out.write_text(json.dumps(entries), encoding="utf-8")

    print()
    print(f"wrote {out}")
    print(f"entries {len(entries)}, boxes {total_boxes}, dropped degenerate {skipped}")
    per_slot = [sum(len(e[s]) for e in entries) for s in range(len(SUBMISSION_ORDER))]
    for slot, count in enumerate(per_slot):
        print(f"  slot {slot} ({SUBMISSION_ORDER[slot]:<13}): {count:>6} boxes")
    if entries:
        print(f"shape per entry: {[len(s) for s in entries[0]][:8]} slots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
