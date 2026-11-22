"""Does fusing our models beat the best single model? Measure before distilling.

    python src/ensemble_eval.py
    python src/ensemble_eval.py --models F,B,H1 --tta

Why this runs before any distillation work: distillation only pays off when the
teacher is meaningfully stronger than the student.  A teacher built by fusing our
own models can only be as strong as the fusion, so the fusion number decides
whether the 4-hour distillation rerun is worth starting.

Everything is scored by the SAME pycocotools COCOeval at IoU 0.5 over the same
489-image leak-free split, so single-model and fused numbers are comparable; the
single-model numbers printed here are re-measured, not copied from other runs.

Fusion is Weighted Box Fusion (Solovyev et al.) implemented here per class:
boxes are clustered by IoU, cluster coordinates are score-weighted means, and the
cluster score is rescaled by how many models voted, so a box only one model
proposed is penalised.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\X光")
VAL_IMG = ROOT / "data" / "yolo_clean" / "images" / "val"
VAL_LBL = ROOT / "data" / "yolo_clean" / "labels" / "val"
VAL_MANIFEST = ROOT / "runs" / "audit" / "E_content_disjoint_val.txt"
OUT = ROOT / "runs" / "audit" / "ensemble"

CLASSES = ["drinkbottle", "glassbottle", "knife", "laptop",
           "metalbottle", "pressuretank", "tool", "umbrella"]


def manifest_stems(path: Path) -> list[str] | None:
    """Image stems listed in the evaluation manifest, or None if unavailable.

    The manifest is the only definition of the 489-image split that every earlier
    number (0.7835, 0.7858, 0.7603) was measured on.  data/yolo_clean/val holds
    552 images, so evaluating all of them would silently produce numbers that
    cannot be compared with anything.  The stored paths contain a mojibake copy of
    the Chinese directory, but the file names themselves are intact, so the stem
    is recovered from the tail of each line.
    """
    if not path.is_file():
        return None
    stems = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        tail = line.replace("\\", "/").rsplit("/", 1)[-1]
        if tail.lower().endswith(".jpg"):
            stems.append(tail[:-4])
    return stems or None

MODELS = {
    "F": ROOT / "runs/first_test/N75_20261120_F1024/weights/best_map50.pt",
    "B": ROOT / "runs/first_test/N75b_cls0625/weights/best_map50.pt",
    "H1": ROOT / "runs/first_test/H1_hgnetv2/weights/best_map50.pt",
    "D": ROOT / "runs/first_test/D_ft1024/weights/best.pt",
    "G": ROOT / "runs/first_test/N75_20261120_G1280/weights/best_map50.pt",
    "R1": ROOT / "runs/first_test/R1_rtdetr1024/weights/best_map50.pt",
    "S1": ROOT / "runs/first_test/S1_semi_yolov8l/weights/best_map50.pt",
    "S2": ROOT / "runs/first_test/S2_semi_hgnetv2/weights/best_map50.pt",
    "S3": ROOT / "runs/first_test/S3_semi_round2/weights/best_map50.pt",
    "S4": ROOT / "runs/first_test/S4_copypaste/weights/best_map50.pt",
}


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
            wpx, hpx = bw * w, bh * h
            anns.append({"id": ann_id, "image_id": img_id,
                         "category_id": CLASSES[int(float(c))],
                         "bbox": [x1, y1, wpx, hpx], "area": wpx * hpx, "iscrowd": 0})
            ann_id += 1
    return images, anns


def score(gt_path: Path, preds: list[dict], label: str, quiet: bool = False,
          out_dir: Path | None = None) -> dict:
    import contextlib as cl
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    target = out_dir or OUT
    target.mkdir(parents=True, exist_ok=True)
    pred_path = target / f"preds_{label}.json"
    pred_path.write_text(json.dumps(preds), encoding="utf-8")
    with cl.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]
    per = {}
    for k, cls in enumerate(CLASSES):
        arr = prec[0, :, k, 0, 2]
        arr = arr[arr > -1]
        per[cls] = float(np.mean(arr)) if arr.size else float("nan")
    mean = float(np.nanmean(list(per.values())))
    if not quiet:
        print(f"    {label:<22} mAP50 = {mean:.4f}")
    return {"label": label, "mAP50": mean, "per_class": per}


def predict(model_path: Path, img_files: list[Path], imgsz: int, tta: bool,
            conf: float) -> dict[int, list[tuple[int, float, list[float]]]]:
    """Return {image_index: [(canonical_class_index, score, [x1,y1,x2,y2]), ...]}.

    Class ids are translated by NAME, not by position.  The previous version built
    `order = [names.index(c) for c in CLASSES]` and then indexed it with the
    model's own class id -- that happens to be correct only while the model's
    names list is in exactly CLASSES order.  With any other ordering (a checkpoint
    whose head was remapped, e.g. the EDS-trained ones) it would silently mislabel
    every box instead of failing.
    """
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    canonical = {name: idx for idx, name in enumerate(CLASSES)}
    unknown = [n for n in names if n not in canonical]
    if unknown:
        raise ValueError(f"{model_path.name}: classes outside this task: {unknown}")

    out: dict[int, list[tuple[int, float, list[float]]]] = {i: [] for i in range(len(img_files))}
    # Batched inference: the per-image loop this replaced spent ~4.5 min per model
    # on 489 images, which made an eight-model evaluation need 36 minutes.  Batch
    # 8 cuts that by roughly 2-3x with identical results.
    BATCH = 8
    for start in range(0, len(img_files), BATCH):
        chunk = img_files[start: start + BATCH]
        with contextlib.redirect_stdout(io.StringIO()):
            results = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=conf,
                                    iou=0.7, max_det=300, augment=tta, rect=False,
                                    verbose=False, device=0)
        for offset, r in enumerate(results):
            idx = start + offset
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            for box, sc, cl in zip(b, s, c):
                mapped = canonical[names[int(cl)]]
                out[idx].append((mapped, float(sc), [float(v) for v in box]))
        if (start + BATCH) % 100 < BATCH:
            print(f"      {min(start + BATCH, len(img_files))}/{len(img_files)}", flush=True)
    return out, names


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def wbf_image(boxes_by_model: list[np.ndarray], scores_by_model: list[np.ndarray],
              weights: list[float], iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Fuse one class of one image across models. boxes (N,4), scores (N,)."""
    allb, alls, allm = [], [], []
    for mi, (b, s) in enumerate(zip(boxes_by_model, scores_by_model)):
        for k in range(len(b)):
            allb.append(b[k])
            alls.append(s[k])
            allm.append(mi)
    if not allb:
        return np.zeros((0, 4)), np.zeros(0)
    allb = np.asarray(allb, dtype=np.float32)
    alls = np.asarray(alls, dtype=np.float32)
    allm = np.asarray(allm, dtype=np.int64)
    order = np.argsort(-alls)
    allb, alls, allm = allb[order], alls[order], allm[order]

    n_models = len(boxes_by_model)
    used = np.zeros(len(alls), dtype=bool)
    fused_b, fused_s = [], []
    for i in range(len(alls)):
        if used[i]:
            continue
        ious = iou_matrix(allb[i: i + 1], allb)[0]
        members = np.where((ious >= iou_thr) & (~used))[0]
        # at most one box per model votes for a cluster
        seen, keep = set(), []
        for j in members:
            if allm[j] in seen:
                continue
            seen.add(allm[j])
            keep.append(j)
        keep = np.asarray(keep, dtype=np.int64)   # empty list would default to float
        if keep.size == 0:
            continue
        used[keep] = True
        w = np.array([weights[allm[j]] for j in keep], dtype=np.float32)
        ws = alls[keep]
        fused_b.append((allb[keep] * (ws * w)[:, None]).sum(0) / np.maximum((ws * w).sum(), 1e-9))
        fused_s.append(float(ws.mean() * min(len(seen), n_models) / n_models))
    return np.asarray(fused_b, dtype=np.float32), np.asarray(fused_s, dtype=np.float32)


def sha256_file(path: Path, limit: int | None = None) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
            read += len(block)
            if limit and read >= limit:
                break
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="F,B,H1")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--iou-grid", default="0.5,0.55,0.6,0.65")
    ap.add_argument("--imgsz-map", default="",
                    help="per-model resolution override, e.g. 'R1=640'.  RT-DETR was "
                         "trained at 640 and evaluating it at 1024 measures a resolution it "
                         "never saw.")
    ap.add_argument("--val-manifest", default=str(VAL_MANIFEST))
    ap.add_argument("--all-val", action="store_true",
                    help="ignore the manifest and score all 552 val images "
                         "(numbers then are NOT comparable to earlier runs)")
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if not MODELS[k].is_file():
            print(f"missing checkpoint for {k}: {MODELS[k]}")
            return 1

    img_files = sorted(VAL_IMG.glob("*.jpg"))
    if not args.all_val:
        stems = manifest_stems(Path(args.val_manifest))
        if stems:
            wanted = set(stems)
            img_files = [p for p in img_files if p.stem in wanted]
            print(f"restricted to manifest: {len(img_files)} of the {len(wanted)} listed")
        else:
            print(f"WARNING: manifest not usable at {args.val_manifest}; "
                  f"scoring all {len(img_files)} images -- not comparable to earlier runs")
    print(f"val images: {len(img_files)}")
    images, anns = load_gt(img_files)

    # Every fusion experiment gets its own directory.  A single shared
    # ensemble_result.json silently overwrote the 3-model result as soon as the
    # 5-model run finished, which is exactly how a number ends up attached to the
    # wrong configuration.
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + "-".join(keys)
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    gt_path = run_dir / "gt.json"
    gt_path.write_text(json.dumps({"images": images, "annotations": anns,
                                   "categories": [{"id": c, "name": c} for c in CLASSES]}),
                       encoding="utf-8")
    print(f"gt boxes: {len(anns)}")
    print(f"run dir : {run_dir}")

    manifest_path = Path(args.val_manifest)
    config = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": keys,
        "checkpoints": {k: str(MODELS[k]) for k in keys},
        "checkpoint_sha256_16": {k: sha256_file(MODELS[k], limit=64 << 20)[:16] for k in keys},
        "imgsz": args.imgsz,
        "imgsz_map": {k: int(dict(part.split("=") for part in args.imgsz_map.split(",") if "=" in part).get(k, args.imgsz)) for k in keys},
        "conf": args.conf,
        "tta": args.tta,
        "iou_grid": args.iou_grid,
        "val_manifest": str(manifest_path),
        "val_manifest_sha256_16": (sha256_file(manifest_path)[:16]
                                   if manifest_path.is_file() else None),
        "n_val_images": len(img_files),
        "n_gt_boxes": len(anns),
        "scorer": "pycocotools COCOeval, iouThr=0.5, all-point",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    per_model = {}
    singles = []
    for k in keys:
        print(f"\n[{k}] predicting ...", flush=True)
        ms = dict(part.split("=") for part in args.imgsz_map.split(",") if "=" in part)
        model_imgsz = int(ms.get(k, args.imgsz))
        if model_imgsz != args.imgsz:
            print(f"  {k}: using its own imgsz {model_imgsz} (global was {args.imgsz})")
        preds, names = predict(MODELS[k], img_files, model_imgsz, args.tta, args.conf)
        print(f"  model classes: {names}")
        per_model[k] = preds
        flat = [{"image_id": i + 1, "category_id": CLASSES[c],
                 "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s}
                for i, lst in preds.items() for (c, s, b) in lst]
        singles.append(score(gt_path, flat, f"single_{k}", out_dir=run_dir))

    weights = {k: s["mAP50"] for k, s in zip(keys, singles)}
    print(f"\n  weights by single-model AP50: "
          f"{ {k: round(v, 4) for k, v in weights.items()} }")

    best = None
    fused_by_iou = []
    for iou_thr in [float(x) for x in args.iou_grid.split(",")]:
        fused = []
        for i in range(len(img_files)):
            for cls in range(len(CLASSES)):
                bb, ss = [], []
                for k in keys:
                    boxes = np.asarray([b for (c, s, b) in per_model[k][i] if c == cls],
                                       dtype=np.float32).reshape(-1, 4)
                    scs = np.asarray([s for (c, s, b) in per_model[k][i] if c == cls],
                                     dtype=np.float32)
                    bb.append(boxes)
                    ss.append(scs)
                fb, fs = wbf_image(bb, ss, [weights[k] for k in keys], iou_thr)
                for box, sc in zip(fb, fs):
                    fused.append({"image_id": i + 1, "category_id": CLASSES[cls],
                                  "bbox": [float(box[0]), float(box[1]),
                                           float(box[2] - box[0]), float(box[3] - box[1])],
                                  "score": float(sc)})
        print(f"\n[fused iou_thr={iou_thr}] {len(fused)} boxes")
        res = score(gt_path, fused, f"fused_iou{int(iou_thr*100)}", out_dir=run_dir)
        res["iou_thr"] = iou_thr
        fused_by_iou.append(res)
        if best is None or res["mAP50"] > best["mAP50"]:
            best = res

    print("\n" + "=" * 62)
    print("summary (same 489-image split, pycocotools AP50)")
    print("=" * 62)
    for s in singles:
        print(f"  single {s['label']:<14} {s['mAP50']:.4f}")
    if best:
        delta = best["mAP50"] - max(s["mAP50"] for s in singles)
        print(f"  FUSED  {best['label']:<14} {best['mAP50']:.4f}   "
              f"({delta:+.4f} vs best single)")

    metrics = {"config": config, "singles": singles,
               "fused_by_iou": fused_by_iou, "best_fused": best}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Handoff file: export_fused reads THIS instead of keeping its own copy of
    # weights and iou_thr.  Two sources of truth is how an export ends up using
    # an IoU threshold the evaluation never blessed.
    fusion_cfg = {
        "run_id": run_id,
        "models": keys,
        "checkpoints": {k: str(MODELS[k]) for k in keys},
        "checkpoint_sha256_16": config["checkpoint_sha256_16"],
        "weights": {k: round(weights[k], 6) for k in keys},
        "iou_thr": best["iou_thr"] if best else 0.5,
        "imgsz": args.imgsz,
        "imgsz_map": config["imgsz_map"],
        "conf": args.conf,
        "tta": args.tta,
        "expected_mAP50": best["mAP50"] if best else None,
        "n_val_images": len(img_files),
        "val_manifest": config["val_manifest"],
    }
    (run_dir / "fusion_config.json").write_text(json.dumps(fusion_cfg, indent=2),
                                                encoding="utf-8")
    print(f"\n-> {run_dir / 'metrics.json'}")
    print(f"-> {run_dir / 'fusion_config.json'}  "
          f"(iou_thr={fusion_cfg['iou_thr']}, export reads this)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


