"""Copy-paste augmentation for the rare classes.

    python src/copy_paste_aug.py --dry-run
    python src/copy_paste_aug.py --copies 2400

Why: the binding constraint is sample count, not tuning.  The whole training set
holds 126 `tool` boxes in 97 images (its validation AP is measured on 12 boxes in 5
images).  No amount of model averaging creates tool examples, so the standard fix
from this competition family (2021 winner: "MixUpObject") is to synthesise them:
cut real instances out of their source images and paste them into other training
images, carrying the box over.

Design decisions and the reason for each:

  * Instances are pasted RECTANGULARLY with a feathered edge.  X-ray images have no
    shadows or lighting to fake, so a plain paste is physically plausible; there is
    nothing for a harmonisation step to fix.
  * Placement rejects positions whose IoU with an existing box exceeds 0.05.  An
    overlapping paste would produce two labels on one object, which is the same
    contradictory-supervision defect we measured earlier in the duplicate audit.
  * Scale jitter 0.75-1.35x, because the audit showed the weak classes are bimodal
    in size (pressuretank: 24% tiny, 20% huge) and a single scale would just
    reinforce the dominant one.
  * Targets are sampled from the SAME dataset the model trains on, and each
    augmented image keeps the target's original labels plus the pasted box, so no
    real annotation is ever lost.
  * Copies are apportioned by scarcity: tool gets the most, since it has 12x fewer
    instances than drinkbottle.

Output is a new dataset directory; source images and labels are never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["drinkbottle", "glassbottle", "knife", "laptop",
           "metalbottle", "pressuretank", "tool", "umbrella"]

ROOT = Path(r"E:\X光")
BASE_IMG = ROOT / "data" / "yolo_semi2" / "images" / "train"
BASE_LBL = ROOT / "data" / "yolo_semi2" / "labels" / "train"
VAL_IMG = ROOT / "data" / "yolo_semi2" / "images" / "val"
VAL_LBL = ROOT / "data" / "yolo_semi2" / "labels" / "val"
OUT = ROOT / "data" / "yolo_cp"

# how many augmented copies to produce, by the class the copy is built around
DEFAULT_PLAN = {"tool": 1100, "pressuretank": 650, "drinkbottle": 650}


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split()
        out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def to_pixels(box, w, h):
    _, cx, cy, bw, bh = box
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None,
                    help="override copy counts, e.g. 'tool=1200,drinkbottle=600'")
    ap.add_argument("--max-iou", type=float, default=0.05)
    ap.add_argument("--max-paste", type=int, default=3, help="instances per augmented image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = dict(DEFAULT_PLAN)
    if args.plan:
        plan = {}
        for part in args.plan.split(","):
            name, _, cnt = part.partition("=")
            plan[name.strip()] = int(cnt)

    rng = random.Random(args.seed)
    base_imgs = sorted(BASE_IMG.glob("*.jpg"))
    if not base_imgs:
        print(f"no base images in {BASE_IMG}")
        return 1

    # ---- collect donor instances for the classes we are going to synthesise ----
    print("collecting donor instances ...", flush=True)
    donors: dict[str, list[tuple[Path, tuple[int, int, int, int]]]] = defaultdict(list)
    for p in base_imgs:
        lbl = BASE_LBL / f"{p.stem}.txt"
        boxes = read_labels(lbl)
        if not boxes:
            continue
        with Image.open(p) as im:
            w, h = im.size
        for b in boxes:
            name = CLASSES[b[0]]
            if name not in plan:
                continue
            x1, y1, x2, y2 = to_pixels(b, w, h)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 - x1 >= 12 and y2 - y1 >= 12:
                donors[name].append((p, (x1, y1, x2, y2)))
    for name in plan:
        print(f"  {name:<14} donors {len(donors[name]):>5}   target copies {plan[name]:>5}")
    if any(not donors[n] for n in plan):
        print("!! a planned class has no usable donors")
        return 1

    out = Path(args.out)
    if args.dry_run:
        print(f"\nDRY RUN: would write {sum(plan.values())} augmented images to {out}")
        return 0
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # base data is hard-linked: only the augmented copies are new bytes
    for p in base_imgs:
        dst = out / "images" / "train" / p.name
        try:
            dst.hardlink_to(p)
        except OSError:
            shutil.copyfile(p, dst)
        src = BASE_LBL / f"{p.stem}.txt"
        if src.exists():
            shutil.copyfile(src, out / "labels" / "train" / f"{p.stem}.txt")
    for p in sorted(VAL_IMG.glob("*.jpg")):
        dst = out / "images" / "val" / p.name
        try:
            dst.hardlink_to(p)
        except OSError:
            shutil.copyfile(p, dst)
        src = VAL_LBL / f"{p.stem}.txt"
        if src.exists():
            shutil.copyfile(src, out / "labels" / "val" / f"{p.stem}.txt")

    # ---- build the augmented copies ----
    made = 0
    stats = Counter()
    rejected = 0
    for cls_name, want in plan.items():
        for i in range(want):
            target = rng.choice(base_imgs)
            with Image.open(target) as im_t:
                canvas = im_t.convert("RGB")
            w, h = canvas.size
            base_boxes = read_labels(BASE_LBL / f"{target.stem}.txt")
            existing = [to_pixels(b, w, h) for b in base_boxes]

            new_boxes = []
            n_paste = rng.randint(1, args.max_paste)
            for _ in range(n_paste):
                donor_path, crop_box = rng.choice(donors[cls_name])
                with Image.open(donor_path) as im_d:
                    patch = im_d.convert("RGB").crop(crop_box)
                scale = rng.uniform(0.75, 1.35)
                pw = max(12, int(patch.width * scale))
                ph = max(12, int(patch.height * scale))
                if pw >= w - 4 or ph >= h - 4:
                    continue
                patch = patch.resize((pw, ph), Image.BILINEAR)

                placed = False
                for _try in range(60):
                    px = rng.randint(0, w - pw - 1)
                    py = rng.randint(0, h - ph - 1)
                    box = (px, py, px + pw, py + ph)
                    if all(iou(box, e) <= args.max_iou for e in existing):
                        # Feathered composite instead of a hard paste.  A pasted
                        # crop carries its own background density, so a sharp
                        # rectangle leaves a visible seam and the detector can learn
                        # to key on the seam rather than the object -- the classic
                        # copy-paste failure mode.
                        mask = Image.new("L", (pw, ph), 0)
                        draw = ImageDraw.Draw(mask)
                        inset = max(2, int(min(pw, ph) * 0.12))
                        draw.rectangle([inset, inset, pw - inset, ph - inset], fill=255)
                        mask = mask.filter(ImageFilter.GaussianBlur(radius=inset * 0.8))
                        canvas.paste(patch, (px, py), mask)
                        new_boxes.append((px, py, pw, ph))
                        existing.append(box)
                        placed = True
                        break
                if not placed:
                    rejected += 1

            if not new_boxes:
                continue
            made += 1
            name = f"cp_{cls_name}_{i:05d}"
            canvas.save(out / "images" / "train" / f"{name}.jpg", quality=92)
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in base_boxes]
            for (px, py, pw, ph) in new_boxes:
                cx, cy = (px + pw / 2) / w, (py + ph / 2) / h
                lines.append(f"{CLASSES.index(cls_name)} {cx:.6f} {cy:.6f} "
                             f"{pw / w:.6f} {ph / h:.6f}")
                stats[cls_name] += 1
            (out / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines) + "\n",
                                                                  encoding="utf-8")

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8")

    report = {
        "base_dataset": str(BASE_IMG.parent.parent),
        "out": str(out),
        "plan": plan,
        "augmented_images": made,
        "pasted_instances": dict(stats),
        "rejected_placements": rejected,
        "placements_rejected_reason": "no position free of overlap (IoU > max_iou)",
        "max_iou": args.max_iou,
        "scale_range": [0.75, 1.35],
        "base_train_images": len(base_imgs),
        "combined_train_images": len(base_imgs) + made,
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\naugmented images: {made}")
    print(f"pasted instances: {dict(stats)}")
    print(f"rejected placements: {rejected}")
    print(f"combined train: {len(base_imgs) + made}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
