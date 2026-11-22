"""Prepare a separate, conservative dataset for an overnight training run.

This only copies the existing supervised training images and their complete
labels. It adds no pseudo-labels, removes no small boxes, and does not resolve
duplicate-image annotation conflicts. Repetition is the maximum TOTAL exposure
requested by any class present in an image, not the sum of class weights.

    python src/prepare_night_dataset.py --dry-run
    python src/prepare_night_dataset.py

The output must not already exist. Source files are never changed or linked to
the output: ordinary copies protect them from later edits in the new dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from classes import CLASSES

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOTAL_REPEAT = {"tool": 3, "drinkbottle": 2, "pressuretank": 2, "knife": 2}


def parse_repeat(text: str) -> dict[str, int]:
    """Parse --repeat 'tool=3,pressuretank=2' into {class: total exposure}.

    Replaces the default table entirely, so a class that is left out gets
    1 exposure. Needed because the first night's run showed the knife and
    drinkbottle multipliers were stealing exposure from tool without helping
    those classes (knife lost 6 AP50 points on the clean validation split).
    """
    out: dict[str, int] = {}
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--repeat entry must look like class=times, got {part!r}")
        name, _, times = part.partition("=")
        name = name.strip()
        if name not in CLASSES:
            raise ValueError(f"unknown class {name!r} in --repeat")
        value = int(times)
        if value < 1:
            raise ValueError(f"repeat for {name} must be >= 1")
        out[name] = value
    return out


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    rows: tuple[tuple[int, float, float, float, float], ...]
    sha256: str


@dataclass
class Plan:
    source: Path
    manifest: Path
    out: Path
    train: list[tuple[Sample, int]]
    val: list[Sample]
    report: dict


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    matches = [i for i, part in enumerate(parts) if part.casefold() == "images"]
    if not matches:
        raise ValueError(f"Cannot derive labels path: no 'images' directory in {image}")
    parts[matches[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_rows(label: Path) -> tuple[tuple[int, float, float, float, float], ...]:
    if not label.is_file():
        raise ValueError(f"Label not found: {label}")
    rows = []
    for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected class/cx/cy/w/h at {label}:{number}")
        try:
            cid_value, cx, cy, width, height = map(float, fields)
        except ValueError as exc:
            raise ValueError(f"Non-numeric label at {label}:{number}") from exc
        if not all(math.isfinite(v) for v in (cid_value, cx, cy, width, height)):
            raise ValueError(f"Non-finite label at {label}:{number}")
        if not cid_value.is_integer() or not 0 <= cid_value < len(CLASSES):
            raise ValueError(f"Invalid class ID at {label}:{number}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid normalized coordinates at {label}:{number}")
        # Converted six-decimal labels can round a border by about 1e-6.
        tolerance = 1e-5
        if (cx - width / 2 < -tolerance or cy - height / 2 < -tolerance
                or cx + width / 2 > 1 + tolerance or cy + height / 2 > 1 + tolerance):
            raise ValueError(f"Box outside image at {label}:{number}")
        rows.append((int(cid_value), cx, cy, width, height))
    return tuple(rows)


def sha256_file(path: Path) -> tuple[Path, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return path, digest.hexdigest()


def read_class_order(source: Path) -> None:
    # PyYAML is already a dependency of the project's Ultralytics installation;
    # this import does not import torch or initialize a GPU.
    import yaml

    dataset_yaml = source / "data.yaml"
    if not dataset_yaml.is_file():
        raise ValueError(f"Source data.yaml not found: {dataset_yaml}")
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8-sig"))
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        try:
            order = [names[i] if i in names else names[str(i)] for i in range(len(CLASSES))]
        except KeyError as exc:
            raise ValueError("Source data.yaml does not define all eight class IDs") from exc
        if len(names) != len(CLASSES):
            raise ValueError("Source data.yaml has an unexpected class count")
    elif isinstance(names, list):
        order = names
    else:
        raise ValueError("Source data.yaml must contain names as a list or ID/name mapping")
    if order != CLASSES:
        raise ValueError(f"Class order differs from src/classes.py: {order!r}")


def read_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise ValueError(f"Validation manifest not found: {path}")
    images = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if not value:
            continue
        image = Path(value).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        images.append(image.resolve())
    if not images:
        raise ValueError("Validation manifest is empty")
    if len({str(p).casefold() for p in images}) != len(images):
        raise ValueError("Validation manifest repeats an image path")
    if len({p.stem.casefold() for p in images}) != len(images):
        raise ValueError("Validation image stems collide in the output")
    return sorted(images)


def duplicate_details(samples: list[Sample]) -> list[dict]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.sha256].append(sample)
    return [
        {
            "sha256": digest,
            "images": [sample.image.as_posix() for sample in members],
            "annotation_sets_identical": len({tuple(sorted(sample.rows)) for sample in members}) == 1,
        }
        for digest, members in groups.items() if len(members) > 1
    ]


def make_plan(source: Path, manifest: Path, out: Path) -> Plan:
    source, manifest, out = source.resolve(), manifest.resolve(), out.resolve()
    read_class_order(source)
    if not (source / "images" / "train").is_dir() or not (source / "images" / "val").is_dir():
        raise ValueError("Source must have images/train and images/val directories")
    train_images = sorted(
        p.resolve() for p in (source / "images" / "train").iterdir()
        if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES
    )
    if not train_images:
        raise ValueError("Source training set is empty")
    if len({p.stem.casefold() for p in train_images}) != len(train_images):
        raise ValueError("Source training image stems collide")
    val_images = read_manifest(manifest)
    for image in train_images + val_images:
        if not image.is_file():
            raise ValueError(f"Image not found: {image}")
        if image.suffix.casefold() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {image}")
    if source.is_relative_to(out) or manifest.is_relative_to(out):
        raise ValueError("Output must not contain the source dataset or validation manifest")

    label_rows = {image: read_rows(label_for_image(image)) for image in train_images + val_images}
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(sha256_file, train_images + val_images))
    samples = {image: Sample(image, label_for_image(image), label_rows[image], hashes[image])
               for image in train_images + val_images}
    train_by_hash: dict[str, list[Path]] = defaultdict(list)
    for image in train_images:
        train_by_hash[hashes[image]].append(image)
    collisions = [
        {"val": image.as_posix(), "train": [p.as_posix() for p in train_by_hash[hashes[image]]]}
        for image in val_images if hashes[image] in train_by_hash
    ]
    if collisions:
        raise ValueError("Training/validation exact-content collision(s): "
                         + json.dumps(collisions[:5], ensure_ascii=True))

    train, destination_names = [], set()
    source_boxes: Counter = Counter()
    effective_boxes: Counter = Counter()
    repeat_counts: Counter = Counter()
    bytes_to_copy = 0
    for image in train_images:
        sample = samples[image]
        present = {CLASSES[row[0]] for row in sample.rows}
        repeat = max((TOTAL_REPEAT.get(name, 1) for name in present), default=1)
        train.append((sample, repeat))
        repeat_counts[repeat] += 1
        for row in sample.rows:
            source_boxes[CLASSES[row[0]]] += 1
            effective_boxes[CLASSES[row[0]]] += repeat
        for copy_number in range(1, repeat + 1):
            name = image.name if copy_number == 1 else f"{image.stem}_r{copy_number}{image.suffix}"
            if name.casefold() in destination_names:
                raise ValueError(f"Generated training filename collision: {name}")
            destination_names.add(name.casefold())
        bytes_to_copy += repeat * (image.stat().st_size + sample.label.stat().st_size)

    val = [samples[image] for image in val_images]
    val_boxes = Counter(CLASSES[row[0]] for sample in val for row in sample.rows)
    missing = [name for name in CLASSES if not val_boxes[name]]
    if missing:
        raise ValueError(f"Frozen validation set has no ground truth for: {missing}")
    bytes_to_copy += sum(sample.image.stat().st_size + sample.label.stat().st_size for sample in val)
    train_duplicates = duplicate_details([sample for sample, _ in train])
    val_duplicates = duplicate_details(val)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source.as_posix(),
        "validation_manifest": manifest.as_posix(),
        "validation_manifest_sha256": sha256_file(manifest)[1],
        "output": out.as_posix(),
        "class_order": CLASSES,
        "total_exposure_by_class": TOTAL_REPEAT,
        "repeat_rule": "maximum total exposure of any class present; never sum class weights",
        "source_training_images": len(train),
        "effective_training_images": sum(repeat for _, repeat in train),
        "source_images_by_repeat_count": dict(sorted(repeat_counts.items())),
        "validation_images": len(val),
        "source_boxes_per_class": {name: source_boxes[name] for name in CLASSES},
        "effective_training_box_exposures": {name: effective_boxes[name] for name in CLASSES},
        "validation_boxes_per_class": {name: val_boxes[name] for name in CLASSES},
        "estimated_copy_bytes": bytes_to_copy,
        "sha256_train_val_collisions": 0,
        "source_train_duplicate_groups": len(train_duplicates),
        "source_train_duplicate_groups_with_annotation_differences": sum(
            not group["annotation_sets_identical"] for group in train_duplicates),
        "source_train_redundant_images": sum(len(group["images"]) - 1 for group in train_duplicates),
        "validation_duplicate_groups": len(val_duplicates),
        "source_train_duplicate_details": train_duplicates,
        "validation_duplicate_details": val_duplicates,
        "policy": {
            "pseudo_labels_added": False,
            "small_boxes_removed": False,
            "automatic_deduplication": False,
            "annotation_conflicts_resolved": False,
            "near_duplicate_scenes_reviewed": False,
            "source_files_modified": False,
            "copy_method": "independent ordinary file copies, no hardlinks",
        },
    }
    return Plan(source, manifest, out, train, val, report)


def write_dataset(plan: Plan) -> None:
    if plan.out.exists():
        raise ValueError(f"Output already exists; choose a new --out: {plan.out}")
    ancestor = plan.out.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    required = int(plan.report["estimated_copy_bytes"] * 1.05) + (1 << 20)
    if shutil.disk_usage(ancestor).free < required:
        raise ValueError(f"Insufficient free disk space; need about {required:,} bytes")
    plan.out.mkdir(parents=True, exist_ok=False)
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (plan.out / sub).mkdir(parents=True)
    for sample, repeat in plan.train:
        for copy_number in range(1, repeat + 1):
            stem = sample.image.stem if copy_number == 1 else f"{sample.image.stem}_r{copy_number}"
            shutil.copy2(sample.image, plan.out / "images/train" / f"{stem}{sample.image.suffix}")
            shutil.copy2(sample.label, plan.out / "labels/train" / f"{stem}.txt")
    for sample in plan.val:
        shutil.copy2(sample.image, plan.out / "images/val" / sample.image.name)
        shutil.copy2(sample.label, plan.out / "labels/val" / f"{sample.image.stem}.txt")
    (plan.out / "data.yaml").write_text(
        f"path: {plan.out.as_posix()}\ntrain: images/train\nval: images/val\nnc: {len(CLASSES)}\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES)),
        encoding="utf-8",
    )
    (plan.out / "prepare_report.json").write_text(
        json.dumps(plan.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/yolo")
    parser.add_argument("--val-manifest", default="runs/audit/E_content_disjoint_val.txt")
    parser.add_argument("--out", default="data/yolo_night75")
    parser.add_argument("--repeat", default=None,
                        help="override per-class total exposure, e.g. 'tool=3,pressuretank=2'")
    parser.add_argument("--dry-run", action="store_true", help="validate and print plan without writing files")
    args = parser.parse_args()
    if args.repeat is not None:
        override = parse_repeat(args.repeat)
        TOTAL_REPEAT.clear()
        TOTAL_REPEAT.update(override)
    try:
        source, manifest, out = map(resolve_path, (args.source, args.val_manifest, args.out))
        if out.exists() and not args.dry_run:
            raise ValueError(f"Output already exists; choose a new --out: {out}")
        plan = make_plan(source, manifest, out)
        summary = {key: value for key, value in plan.report.items()
                   if key not in {"source_train_duplicate_details", "validation_duplicate_details"}}
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        if args.dry_run:
            print("DRY RUN: no dataset files were written.")
        else:
            write_dataset(plan)
            print(f"Dataset prepared: {out / 'data.yaml'}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Preparation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
