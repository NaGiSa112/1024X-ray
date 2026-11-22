"""Convert the VOC annotations into a YOLO dataset with a stratified split.

    python src/prepare_dataset.py
    python src/prepare_dataset.py --val-frac 0.15 --seed 1 --clean

What it produces (all under data/):

    data/yolo/images/{train,val}/*.jpg     image files (hard-linked, no copy)
    data/yolo/labels/{train,val}/*.txt     YOLO label files
    data/yolo/data.yaml                    ready for ultralytics
    data/annotations.csv                   every box, one row each
    data/dataset_summary.json              all numbers, machine readable

and prints the class statistics: boxes/images per class, per split, plus the
submission-template view of the test set.

The split is grouped by multi-label signature ("which classes appear together")
so that rare classes such as tool and knife are spread over both splits, and a
final pass forces every class to appear in val at least once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES, NAME_TO_ID, NUM_CLASSES, ZH_NAME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TRAIN_XML = TRAIN_DIR / "XML"
TEST_DIR = ROOT / "data" / "test2"
SUBMISSION = ROOT / "results_test2.json"

OUT = ROOT / "data" / "yolo"
CSV_PATH = ROOT / "data" / "annotations.csv"
SUMMARY_PATH = ROOT / "data" / "dataset_summary.json"
REPORT_PATH = ROOT / "data" / "dataset_report.md"

MIN_BOX_PX = 1.0       # drop boxes thinner than this after clamping
MIN_BOX_NORM = 1e-4    # ... and boxes that are degenerate after normalisation


def parse_voc(xml_path: Path) -> tuple[str, int, int, list[tuple[str, float, float, float, float]], list[str]]:
    """Return (stem, width, height, boxes, warnings)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    warnings: list[str] = []
    boxes: list[tuple[str, float, float, float, float]] = []

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in NAME_TO_ID:
            warnings.append(f"unknown class '{name}'")
            continue
        box = obj.find("bndbox")
        if box is None:
            warnings.append(f"{name}: no bndbox")
            continue
        try:
            x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            warnings.append(f"{name}: unparsable coordinates")
            continue

        # order, then clamp into the image
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        cx1, cx2 = max(0.0, x1), min(float(width), x2)
        cy1, cy2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            warnings.append(f"{name}: degenerate box {x1, y1, x2, y2}")
            continue
        if (cx2 - cx1) < MIN_BOX_PX or (cy2 - cy1) < MIN_BOX_PX:
            warnings.append(f"{name}: box outside image {x1, y1, x2, y2}")
            continue
        if (cx2 - cx1) != (x2 - x1) or (cy2 - cy1) != (y2 - y1):
            warnings.append(f"{name}: clipped to image bounds")
        boxes.append((name, cx1, cy1, cx2, cy2))

    return Path(root.findtext("filename") or xml_path.stem).stem, width, height, boxes, warnings


def stratified_split(
    records: list[dict], val_frac: float, seed: int
) -> tuple[set[str], set[str]]:
    """Group by multi-label signature, split each group, then fix val coverage."""
    groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for rec in records:
        groups[frozenset(rec["classes"])].append(rec["stem"])

    rng = random.Random(seed)
    val: set[str] = set()
    for signature, stems in sorted(groups.items(), key=lambda kv: (len(kv[0]), sorted(kv[0]))):
        stems = sorted(stems)
        rng.shuffle(stems)
        n_val = int(round(len(stems) * val_frac))
        if len(stems) >= 5 and n_val == 0:
            n_val = 1                      # keep rare combinations represented
        val.update(stems[:n_val])

    train = {r["stem"] for r in records} - val

    # force every class to appear in val at least once
    by_stem = {r["stem"]: r for r in records}
    val_classes = {c for s in val for c in by_stem[s]["classes"]}
    for cls in CLASSES:
        if cls in val_classes:
            continue
        candidates = [s for s in train if cls in by_stem[s]["classes"]]
        if not candidates:
            continue
        # move the image whose other classes are best covered in val already
        candidates.sort(key=lambda s: -len(set(by_stem[s]["classes"]) & val_classes))
        pick = candidates[0]
        train.discard(pick)
        val.add(pick)
        val_classes = {c for s in val for c in by_stem[s]["classes"]}

    return train, val


def link_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)          # same volume: no extra disk, no copy time
    except OSError:
        shutil.copyfile(src, dst)


def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def describe(values: list[int]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)

    def q(p: float) -> int:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "min": s[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": s[-1],
        "mean": round(sum(s) / n, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.10, help="fraction of images used for val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="wipe data/yolo before writing")
    args = ap.parse_args()

    report: list[str] = []

    def emit(line: str = "") -> None:
        """Print and remember, so the same text also lands in dataset_report.md."""
        print(line)
        report.append(line)

    if not TRAIN_XML.is_dir():
        print(f"annotations not found: {TRAIN_XML}")
        return 1
    if args.clean and OUT.exists():
        shutil.rmtree(OUT)

    # ---------------- parse everything ----------------
    xmls = sorted(TRAIN_XML.glob("*.xml"))
    records: list[dict] = []
    all_warnings: list[str] = []
    missing_images: list[str] = []

    for xml_path in xmls:
        stem, width, height, boxes, warns = parse_voc(xml_path)
        img = TRAIN_DIR / f"{stem}.jpg"
        if not img.exists():
            missing_images.append(f"{stem}.jpg")
            continue
        all_warnings.extend(f"{xml_path.name}: {w}" for w in warns)
        records.append(
            {
                "stem": stem,
                "image": img,
                "width": width,
                "height": height,
                "boxes": boxes,
                "classes": sorted({b[0] for b in boxes}),
            }
        )

    total_boxes = sum(len(r["boxes"]) for r in records)
    print(f"parsed {len(records)} annotated images, {total_boxes} boxes")
    if missing_images:
        print(f"!! {len(missing_images)} annotations reference a missing image")
    if all_warnings:
        print(f"!! {len(all_warnings)} annotation warnings (first 5):")
        for w in all_warnings[:5]:
            print(f"   {w}")

    # ---------------- split ----------------
    train_stems, val_stems = stratified_split(records, args.val_frac, args.seed)
    print(f"split: train {len(train_stems)} images / val {len(val_stems)} images "
          f"(val_frac={args.val_frac}, seed={args.seed})")

    # ---------------- write YOLO tree ----------------
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    csv_rows: list[list] = []
    per_class_split = {
        "train": {"boxes": Counter(), "images": Counter()},
        "val": {"boxes": Counter(), "images": Counter()},
    }

    for rec in records:
        split = "val" if rec["stem"] in val_stems else "train"
        link_image(rec["image"], OUT / "images" / split / rec["image"].name)

        lines = []
        for name, x1, y1, x2, y2 in rec["boxes"]:
            w, h = rec["width"], rec["height"]
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw < MIN_BOX_NORM or bh < MIN_BOX_NORM:
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"{NAME_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            csv_rows.append(
                [rec["stem"], w, h, name, NAME_TO_ID[name], round(x1, 2), round(y1, 2),
                 round(x2, 2), round(y2, 2), round(bw * bh, 6), split]
            )
            per_class_split[split]["boxes"][name] += 1
        for name in rec["classes"]:
            if any(l.startswith(f"{NAME_TO_ID[name]} ") for l in lines):
                per_class_split[split]["images"][name] += 1

        (OUT / "labels" / split / f"{rec['stem']}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    # data.yaml -- absolute, POSIX separators, UTF-8 (the path contains 中文)
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {NUM_CLASSES}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8",
    )

    # ---------------- csv ----------------
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "width", "height", "class", "class_id",
                         "x1", "y1", "x2", "y2", "area_norm", "split"])
        writer.writerows(csv_rows)

    # ---------------- statistics ----------------
    total_images = {"train": len(train_stems), "val": len(val_stems)}
    per_class_total_boxes = Counter()
    per_class_total_images = Counter()
    for rec in records:
        for name, *_ in rec["boxes"]:
            per_class_total_boxes[name] += 1
        for name in rec["classes"]:
            per_class_total_images[name] += 1

    boxes_per_image = [len(r["boxes"]) for r in records]
    multi = sum(1 for r in records if len(r["classes"]) > 1)
    sizes = [f"{r['width']}x{r['height']}" for r in records]
    size_counter = Counter(sizes)
    areas = sorted(r["width"] * r["height"] for r in records)
    aspect = [r["width"] / r["height"] for r in records]

    emit()
    emit("## 训练集类别数量（按框数降序）")
    emit()
    emit("| id | class | 中文 | 框数 | 占比 | 图片数 | train 框 | val 框 |")
    emit("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, count in per_class_total_boxes.most_common():
        emit(f"| {NAME_TO_ID[name]} | {name} | {ZH_NAME.get(name, '')} | {count} | "
             f"{pct(count, total_boxes)} | {per_class_total_images[name]} | "
             f"{per_class_split['train']['boxes'][name]} | "
             f"{per_class_split['val']['boxes'][name]} |")
    emit(f"| | **TOTAL** | | **{total_boxes}** | 100.0% | **{len(records)}** | "
         f"**{sum(per_class_split['train']['boxes'].values())}** | "
         f"**{sum(per_class_split['val']['boxes'].values())}** |")

    bpi = describe(boxes_per_image)
    asp = describe([round(a, 3) for a in aspect])
    emit()
    emit(f"- 每图框数：min {bpi['min']} / p25 {bpi['p25']} / 中位 {bpi['median']} / "
         f"p75 {bpi['p75']} / p95 {bpi['p95']} / max {bpi['max']} / 均值 {bpi['mean']}")
    emit(f"- 长宽比：min {asp['min']} / 中位 {asp['median']} / max {asp['max']}")
    emit(f"- 图片尺寸：{len(size_counter)} 种不同尺寸（{min(areas):,} ~ {max(areas):,} px）")
    emit(f"- 多类别图片：{multi} 张（{pct(multi, len(records))}）")
    emit(f"- 划分：train {len(train_stems)} 张 / val {len(val_stems)} 张 "
         f"(val_frac={args.val_frac}, seed={args.seed})")

    # ---------------- test set view ----------------
    test_images = sorted(TEST_DIR.glob("*.jpg"))
    submission_stats = None
    if SUBMISSION.exists():
        payload = json.loads(SUBMISSION.read_text(encoding="utf-8"))
        slot_boxes = [0] * len(payload[0])
        empty_slots = 0
        imgs_with_any = 0
        for entry in payload:
            if any(len(slot) for slot in entry):
                imgs_with_any += 1
            for i, slot in enumerate(entry):
                slot_boxes[i] += len(slot)
                if not slot:
                    empty_slots += 1
        submission_stats = {
            "entries": len(payload),
            "classes_per_entry": len(payload[0]),
            "boxes_per_slot": slot_boxes,
            "images_with_predictions": imgs_with_any,
            "empty_slots": empty_slots,
        }

    emit()
    emit("## 测试集")
    emit()
    emit(f"- 图片数：{len(test_images)}")
    emit("- 标注文件：**0** —— 官方没有给测试集标注，所以「测试集每个类有多少个目标」"
         "在数据里查不到，任何具体数字都只能是猜测。")
    if submission_stats:
        emit(f"- 提交模板：{submission_stats['entries']} 条 × "
             f"{submission_stats['classes_per_entry']} 个类别槽位，"
             f"{submission_stats['images_with_predictions']} 张图里有预测框")
        emit("- 模板各槽位的框数（**这是 baseline 模型的预测，不是真值**，只能当参考）：")
        emit()
        emit("| 槽位 | 框数 |")
        emit("| ---: | ---: |")
        for i, count in enumerate(submission_stats["boxes_per_slot"]):
            emit(f"| {i} | {count} |")

    # ---------------- json ----------------
    summary = {
        "classes": CLASSES,
        "class_id": NAME_TO_ID,
        "counts": {
            "train_images": len(records),
            "train_boxes": total_boxes,
            "test_images": len(test_images),
            "test_annotations": 0,
            "split": {"train": total_images["train"], "val": total_images["val"],
                      "val_frac": args.val_frac, "seed": args.seed},
        },
        "per_class": {
            name: {
                "boxes": per_class_total_boxes[name],
                "images": per_class_total_images[name],
                "boxes_train": per_class_split["train"]["boxes"][name],
                "boxes_val": per_class_split["val"]["boxes"][name],
                "images_train": per_class_split["train"]["images"][name],
                "images_val": per_class_split["val"]["images"][name],
            }
            for name in CLASSES
        },
        "boxes_per_image": bpi,
        "aspect_ratio": asp,
        "image_sizes_distinct": len(size_counter),
        "image_sizes_top5": size_counter.most_common(5),
        "multi_class_images": multi,
        "annotation_warnings": len(all_warnings),
        "submission_template": submission_stats,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    emit()
    emit(f"- YOLO 数据集：`{OUT}`（images/labels 的 train+val，data.yaml）")
    emit(f"- 标注表：`{CSV_PATH}`（{len(csv_rows)} 行）")
    emit(f"- 统计 JSON：`{SUMMARY_PATH}`")

    REPORT_PATH.write_text(
        "# 数据集处理报告\n\n"
        f"由 `src/prepare_dataset.py` 生成（val_frac={args.val_frac}, seed={args.seed}）。\n\n"
        + "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
