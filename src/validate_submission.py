"""Validate a submission file against the official template and the test images.

    python src/validate_submission.py results_test2_submit_A.json

Checks, in order:
  * the number of entries equals the number of test images (and the template)
  * every entry has exactly 8 class slots
  * every box is [x1, y1, x2, y2, score] with finite numbers
  * boxes lie inside the image, x1 < x2, y1 < y2, 0 < score <= 1
  * reports boxes per slot and the score distribution

Exit code 0 means the file is structurally safe to submit.  A format error here
is a zero on the leaderboard, so run this on every version.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import SUBMISSION_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test2"
TEMPLATE = ROOT / "results_test2.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", nargs="?", default=str(ROOT / "results_test2_submit.json"))
    ap.add_argument("--max-report", type=int, default=10)
    args = ap.parse_args()

    path = Path(args.submission)
    if not path.exists():
        print(f"not found: {path}")
        return 1

    images = sorted(TEST_DIR.glob("*.jpg"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    n_slots = len(SUBMISSION_ORDER)
    problems: list[str] = []

    print(f"file      : {path}  ({path.stat().st_size / 1024**2:.1f} MiB)")
    print(f"entries   : {len(payload)}   test images: {len(images)}")
    if len(payload) != len(images):
        problems.append(f"entry count {len(payload)} != test images {len(images)}")
    if TEMPLATE.exists():
        n_tpl = len(json.loads(TEMPLATE.read_text(encoding="utf-8")))
        if len(payload) != n_tpl:
            problems.append(f"entry count {len(payload)} != template {n_tpl}")
        print(f"template  : {n_tpl} entries")

    # ---- per-entry structural checks -----------------------------------
    from PIL import Image

    slot_boxes = [0] * n_slots
    scores: list[float] = []
    boxes_per_image: list[int] = []
    out_of_bounds = 0
    bad_shape = 0

    limit = min(len(payload), len(images))
    sizes: dict[int, tuple[int, int]] = {}
    for i in range(limit):
        entry = payload[i]
        if not isinstance(entry, list) or len(entry) != n_slots:
            bad_shape += 1
            if len(problems) < args.max_report:
                problems.append(f"entry {i}: expected {n_slots} slots, got {len(entry) if isinstance(entry, list) else type(entry).__name__}")
            continue
        if i not in sizes:
            with Image.open(images[i]) as im:
                sizes[i] = im.size
        w, h = sizes[i]
        total = 0
        for slot, boxes in enumerate(entry):
            for box in boxes:
                total += 1
                slot_boxes[slot] += 1
                if len(box) != 5:
                    bad_shape += 1
                    continue
                x1, y1, x2, y2, s = box
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                    bad_shape += 1
                    continue
                scores.append(float(s))
                if not (0.0 < s <= 1.0):
                    bad_shape += 1
                if x2 <= x1 or y2 <= y1:
                    bad_shape += 1
                if x1 < 0 or y1 < 0 or x2 > w + 1 or y2 > h + 1:
                    out_of_bounds += 1
        boxes_per_image.append(total)

    print()
    print(f"{'slot':>4}  {'class':<14}{'boxes':>8}")
    for slot, name in enumerate(SUBMISSION_ORDER):
        print(f"{slot:>4}  {name:<14}{slot_boxes[slot]:>8}")
    print(f"      {'TOTAL':<14}{sum(slot_boxes):>8}")

    if boxes_per_image:
        bpi = sorted(boxes_per_image)
        print()
        print(f"boxes/image: min {bpi[0]}  median {bpi[len(bpi) // 2]}  "
              f"mean {sum(bpi) / len(bpi):.1f}  max {bpi[-1]}")
    if scores:
        s = sorted(scores)
        def q(p: float) -> float:
            return s[min(len(s) - 1, int(p * (len(s) - 1)))]
        print(f"scores     : p10 {q(0.10):.4f}  median {q(0.5):.4f}  "
              f"p90 {q(0.90):.4f}  max {s[-1]:.4f}")

    print()
    if bad_shape:
        problems.append(f"{bad_shape} malformed boxes / entries")
    if out_of_bounds:
        problems.append(f"{out_of_bounds} boxes outside the image bounds")

    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems[:args.max_report]:
            print(f"  - {p}")
        return 1

    print("OK — structurally valid submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
