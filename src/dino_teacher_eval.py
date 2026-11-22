"""Measure how good the jgen DINO (EDS-1) teacher really is on THIS task's val set.

    E:\\conda_envs\\dino_teacher\\python.exe src\\dino_teacher_eval.py
    E:\\conda_envs\\dino_teacher\\python.exe src\\dino_teacher_eval.py --limit 60

A teacher is only useful for distillation if it is clearly stronger than the
student on the student's own data.  Absolute paper numbers from another X-ray
dataset do not prove that, so this script runs the checkpoint on our 489-image
leak-free validation split and reports AP50 with pycocotools (the same IoU=0.5,
all-point interpolation the competition uses).

Class coverage: EDS-1 has 10 classes, seven of ours have a counterpart.  metalbottle
has none, so the score here is the mean over the SEVEN mapped classes and is NOT
directly comparable to the 0.786 we get over all eight.  Both numbers are printed
for honesty.

    EDS-1                  -> ours
    knife                    knife
    laptop                   laptop
    umbrella                 umbrella
    glassbottle              glassbottle
    pressure                 pressuretank
    plasticbottle            drinkbottle
    scissor + device         tool   (merged)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"E:\X光")
CFG = ROOT / "deps" / "jgen_repo" / "dino" / "cfg" / "dino-5scale-swin-b-8xb2-36e_EDS_1_2.py"
CKPT = ROOT / "weights" / "dino_eds1.pth"
VAL_IMG = ROOT / "data" / "yolo_clean" / "images" / "val"
VAL_LBL = ROOT / "data" / "yolo_clean" / "labels" / "val"
OUT = ROOT / "runs" / "audit" / "dino_teacher"

CLASSES = ["drinkbottle", "glassbottle", "knife", "laptop",
           "metalbottle", "pressuretank", "tool", "umbrella"]
EDS2OURS = {"knife": "knife", "laptop": "laptop", "umbrella": "umbrella",
            "glassbottle": "glassbottle", "pressure": "pressuretank",
            "plasticbottle": "drinkbottle", "scissor": "tool", "device": "tool"}
EDS_CLASSES = ["plasticbottle", "knife", "scissor", "laptop", "umbrella",
               "lighter", "device", "powerbank", "pressure", "glassbottle"]


def load_gt(img_files: list[Path]) -> tuple[list[dict], list[dict]]:
    from PIL import Image
    images, anns = [], []
    ann_id = 1
    for img_id, p in enumerate(img_files, start=1):
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lbl = VAL_LBL / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            wpx, hpx = (x2 - x1), (y2 - y1)
            anns.append({"id": ann_id, "image_id": img_id,
                         "category_id": CLASSES[int(float(c))] ,
                         "bbox": [x1, y1, wpx, hpx], "area": wpx * hpx, "iscrowd": 0})
            ann_id += 1
    return images, anns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    ap.add_argument("--score-thr", type=float, default=0.01)
    args = ap.parse_args()

    import numpy as np
    import torch
    from mmdet.apis import init_detector, inference_detector
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    img_files = sorted(VAL_IMG.glob("*.jpg"))
    if args.limit:
        img_files = img_files[: args.limit]
    print(f"val images: {len(img_files)}")

    images, anns = load_gt(img_files)
    print(f"gt boxes  : {len(anns)}")

    print("loading DINO ...", flush=True)
    model = init_detector(str(CFG), str(CKPT), device="cuda:0", palette="none")

    results, keep_eds = [], defaultdict(int)
    for i, p in enumerate(img_files, start=1):
        with contextlib.redirect_stdout(io.StringIO()):
            res = inference_detector(model, str(p))
        pred = res.pred_instances
        boxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()
        for (x1, y1, x2, y2), sc, lab in zip(boxes, scores, labels):
            if sc < args.score_thr:
                continue
            eds = EDS_CLASSES[int(lab)] if int(lab) < len(EDS_CLASSES) else None
            ours = EDS2OURS.get(eds) if eds else None
            if ours is None:
                continue
            keep_eds[eds] += 1
            results.append({"image_id": i, "category_id": ours,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(sc)})
        if i % 50 == 0:
            print(f"  {i}/{len(img_files)} images, {len(results)} kept detections", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / "dino_preds.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    print(f"kept detections: {len(results)} -> {pred_path.name}")
    print("  per EDS class:", dict(keep_eds))

    mapped = sorted({v for v in EDS2OURS.values()})
    cats = [{"id": c, "name": c, "supercategory": "object"} for c in mapped]
    gt = {"images": images, "annotations": anns, "categories": cats}
    gt_path = OUT / "dino_gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
    for iou_type in ("bbox",):
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
        with contextlib.redirect_stdout(io.StringIO()):
            ev.summarize()

        prec = ev.eval["precision"]      # [T, R, K, A, M]
        per_class = {}
        for k, cat in enumerate(mapped):
            arr = prec[0, :, k, 0, 2]
            arr = arr[arr > -1]
            per_class[cat] = float(np.mean(arr)) if arr.size else float("nan")

        print("\n=== DINO (EDS-1) on our 489-image leak-free val, AP50 ===")
        print(f"{'class':<14}{'AP50':>9}")
        for c in CLASSES:
            if c in per_class:
                print(f"{c:<14}{per_class[c]:>9.4f}")
            else:
                print(f"{c:<14}{'n/a':>9}")
        vals = [v for v in per_class.values() if v == v]
        mean7 = sum(vals) / len(vals) if vals else float("nan")
        mean8 = sum(vals) / 8 if vals else float("nan")
        print(f"\nmean over {len(vals)} mapped classes : {mean7:.4f}")
        print(f"mean over all 8 (metalbottle=0)  : {mean8:.4f}")
        print(f"our current student (YOLOv8l)    : 0.7858 for reference")

        (OUT / "dino_eval.json").write_text(json.dumps(
            {"per_class": per_class, "mean_mapped": mean7, "mean_all8": mean8,
             "student_reference": 0.7858, "n_images": len(img_files),
             "n_gt_boxes": len(anns), "n_detections": len(results),
             "score_thr": args.score_thr}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
