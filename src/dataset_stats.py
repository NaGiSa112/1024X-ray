"""Inventory the X-ray detection dataset.

    python src/dataset_stats.py

Reads the VOC annotations in data/train/XML and reports class counts, boxes per
image, image sizes and annotation sanity problems.  Also checks that every
image referenced by an annotation exists, and that test2 has the number of
images the submission template expects.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG = ROOT / "data" / "train"
TRAIN_XML = TRAIN_IMG / "XML"
TEST_IMG = ROOT / "data" / "test2"
SUBMISSION = ROOT / "results_test2.json"


def main() -> int:
    if not TRAIN_XML.is_dir():
        print(f"annotation folder not found: {TRAIN_XML}")
        print("extract train.zip first (see README.md)")
        return 1

    xmls = sorted(TRAIN_XML.glob("*.xml"))
    print(f"train annotations : {len(xmls)}")
    print(f"train images      : {len(list(TRAIN_IMG.glob('*.jpg')))}")
    print(f"test2 images      : {len(list(TEST_IMG.glob('*.jpg')))}")

    class_boxes: Counter[str] = Counter()
    class_images: Counter[str] = Counter()
    per_image: list[int] = []
    sizes: Counter[tuple[int, int]] = Counter()
    problems: list[str] = []
    n_boxes = 0

    for xml_path in xmls:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            problems.append(f"{xml_path.name}: unreadable XML ({exc})")
            continue

        size = root.find("size")
        if size is not None:
            w, h = int(size.findtext("width", 0)), int(size.findtext("height", 0))
            sizes[(w, h)] += 1

        stem = Path(root.findtext("filename") or xml_path.stem).stem
        if not (TRAIN_IMG / f"{stem}.jpg").exists():
            problems.append(f"{xml_path.name}: image {stem}.jpg missing")

        names_in_image: set[str] = set()
        objects = root.findall("object")
        per_image.append(len(objects))
        for obj in objects:
            name = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")
            if not name or box is None:
                problems.append(f"{xml_path.name}: object without name/bndbox")
                continue
            try:
                x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError):
                problems.append(f"{xml_path.name}: bad coordinates for {name}")
                continue
            if x2 <= x1 or y2 <= y1:
                problems.append(f"{xml_path.name}: degenerate box for {name} ({x1},{y1},{x2},{y2})")
            class_boxes[name] += 1
            names_in_image.add(name)
            n_boxes += 1
        for name in names_in_image:
            class_images[name] += 1

    print("\nclasses")
    print(f"  {'class':<14}{'boxes':>8}{'images':>8}")
    for name, count in class_boxes.most_common():
        print(f"  {name:<14}{count:>8}{class_images[name]:>8}")
    print(f"  {'TOTAL':<14}{n_boxes:>8}")

    print("\nboxes per image")
    if per_image:
        per_image.sort()
        print(f"  min {per_image[0]} | median {per_image[len(per_image) // 2]} | "
              f"max {per_image[-1]} | mean {n_boxes / len(per_image):.2f}")

    print("\nmost common image sizes")
    for (w, h), count in sizes.most_common(5):
        print(f"  {w}x{h}: {count}")

    print(f"\nproblems: {len(problems)}")
    for line in problems[:20]:
        print(f"  {line}")
    if len(problems) > 20:
        print(f"  ... {len(problems) - 20} more")

    # ---- submission template -------------------------------------------
    if SUBMISSION.exists():
        payload = json.loads(SUBMISSION.read_text(encoding="utf-8"))
        n_test = len(list(TEST_IMG.glob("*.jpg")))
        print("\nsubmission template")
        print(f"  results_test2.json entries : {len(payload)}")
        print(f"  test2 images               : {n_test}")
        print(f"  classes per entry          : {len(payload[0]) if payload else 0}")
        if len(payload) != n_test:
            print("  !! entry count does not match the number of test images")
        print("  -> each entry is a list per class; each box is [x1, y1, x2, y2, score]")
        print("     in original image pixels, classes ordered as in the template.")

    print("\nclass order in the submission template is positional and cannot be")
    print("derived from the data -- confirm it against the competition page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
