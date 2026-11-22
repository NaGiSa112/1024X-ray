"""Fuse several YOLO models with weighted box fusion, and score the result on val.

⚠️ NOT TRUSTWORTHY FOR DECISIONS AS OF 2026-11-20 — read this first.

The AP implementation below does not reproduce Ultralytics on a single model
(D@1024: 0.7485 here vs 0.7587 in `model.val`), and it violates a property the
real metric has: lowering the confidence floor must never lower AP, yet this
code reports 0.7485 at conf=0.001 and 0.7800 at conf=0.05, while Ultralytics
reports 0.7587 at conf=0.001 and 0.7315 at conf=0.05 (i.e. the opposite trend).
So matching/interpolation here still differs from the reference.

Every conclusion drawn from this script before that check was made — the
"raise the export threshold for free AP" idea and the per-class fusion deltas —
is therefore void.  Use `src/evaluate.py` (Ultralytics `model.val`) for anything
that informs a submission.

The fusion code itself (WBF) is independent of the scoring; only the `--eval`
numbers are suspect.  Do not use `--eval` to choose between candidates until the
AP gap is understood and fixed.

    # measure on the validation split (prints mAP@0.5, the official metric)
    python src/fuse_run.py --weights runs/first_test/B_eds640/weights/best.pt \\
        runs/first_test/C_eds_aug640/weights/best.pt \\
        runs/first_test/D_ft1024/weights/best.pt \\
        --imgsz 1024 --eval

    # produce a submission for test2
    python src/fuse_run.py --weights ... --imgsz 1024 --out results_test2_submit_ENS.json

Why this exists: the leaderboard metric is mAP at IoU 0.5 only (per the official
description), and until now we could only measure single models via Ultralytics.
Fusing needs its own evaluation, otherwise an ensemble can only be shipped blind.

The AP implementation follows the official wording: greedy IoU>=0.5 matching,
predictions sorted by confidence, PR curve, all-point interpolated AP, then a
plain mean over the eight classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES, NAME_TO_SLOT, SUBMISSION_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VAL_IMG = ROOT / "data" / "yolo" / "images" / "val"
VAL_LBL = ROOT / "data" / "yolo" / "labels" / "val"
TEST_IMG = ROOT / "data" / "test2"


# --------------------------------------------------------------------- geometry
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------------ WBF
def weighted_box_fusion(boxes_per_model: list[list[list[float]]],
                        weights: list[float],
                        iou_thr: float = 0.55) -> list[list[float]]:
    """Fuse one class for one image.  Each box is [x1, y1, x2, y2, score]."""
    items = []
    for m, boxes in enumerate(boxes_per_model):
        for b in boxes:
            items.append((float(b[4]), m, b))
    items.sort(key=lambda t: -t[0])

    clusters: list[dict] = []
    for score, m, box in items:
        target = -1
        best_iou = iou_thr
        for ci, cl in enumerate(clusters):
            i = iou_xyxy(np.asarray(box[:4], dtype=float), cl["fused"])
            if i > best_iou:
                best_iou, target = i, ci
        if target >= 0:
            clusters[target]["members"].append((score, m, box))
            mem = clusters[target]["members"]
            num = np.zeros(4)
            den = 0.0
            for s, mm, bb in mem:
                w = s * weights[mm]
                num += w * np.asarray(bb[:4], dtype=float)
                den += w
            clusters[target]["fused"] = num / den if den > 0 else clusters[target]["fused"]
        else:
            clusters.append({"members": [(score, m, box)],
                             "fused": np.asarray(box[:4], dtype=float)})

    n_models = len(boxes_per_model)
    out = []
    for cl in clusters:
        mem = cl["members"]
        wsum = sum(s * weights[m] for s, m, _ in mem)
        wden = sum(weights[m] for _, m, _ in mem)
        score = (wsum / wden) if wden > 0 else 0.0
        score *= min(len(mem), n_models) / n_models     # penalise partial agreement
        x1, y1, x2, y2 = cl["fused"]
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append([float(x1), float(y1), float(x2), float(y2), float(score)])
    out.sort(key=lambda b: -b[4])
    return out


# ------------------------------------------------------------------ inference
def run_models(weights: list[str], images: list[Path], imgsz_list: list[int], conf: float,
               iou: float, max_det: int, tta: bool, batch: int, device: str,
               progress_every: int = 200) -> list[list[list[list[float]]]]:
    """Return per image: 8 slots (submission order) of fused boxes.

    Each model keeps its own inference size: a model trained at 640 must be
    evaluated at 640 (we measured C@1024 losing 3.8 mAP50 to the scale mismatch),
    so fusing a 640 model with a 1024 model requires per-model imgsz.
    """
    from ultralytics import YOLO

    models = []
    for w in weights:
        m = YOLO(w)
        unknown = [n for n in m.names.values() if n not in NAME_TO_SLOT]
        if unknown:
            raise SystemExit(f"{w}: classes not in the eight-class set: {unknown}")
        models.append(m)
        print(f"loaded {w}  ({sum(p.numel() for p in m.model.parameters()) / 1e6:.2f}M params)")

    weights_m = [1.0] * len(models)
    entries: list[list[list[list[float]]]] = []

    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start:start + batch]]
        per_model: list[list[list[list[list[float]]]]] = []
        for mi, m in enumerate(models):
            res = m.predict(chunk, imgsz=imgsz_list[mi], conf=conf, iou=iou, max_det=max_det,
                            device=device, augment=tta, verbose=False)
            model_slots = []
            for r in res:
                h, w = r.orig_shape[:2]
                slots: list[list[list[float]]] = [[] for _ in range(len(SUBMISSION_ORDER))]
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    sc = r.boxes.conf.cpu().numpy()
                    cl = r.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), s, c in zip(xyxy, sc, cl):
                        x1 = float(min(max(x1, 0.0), w)); x2 = float(min(max(x2, 0.0), w))
                        y1 = float(min(max(y1, 0.0), h)); y2 = float(min(max(y2, 0.0), h))
                        if x2 - x1 <= 0 or y2 - y1 <= 0:
                            continue
                        slots[NAME_TO_SLOT[m.names[int(c)]]].append([x1, y1, x2, y2, float(s)])
                model_slots.append(slots)
            per_model.append(model_slots)

        for i in range(len(chunk)):
            entry = []
            for slot in range(len(SUBMISSION_ORDER)):
                fused = weighted_box_fusion(
                    [per_model[mi][i][slot] for mi in range(len(models))], weights_m)
                entry.append(fused)
            entries.append(entry)

        done = min(start + batch, len(images))
        if done % progress_every < batch or done == len(images):
            print(f"  {done}/{len(images)} images", flush=True)

    return entries


# ------------------------------------------------------------------ evaluation
def load_gt(labels_dir: Path, images: list[Path]) -> list[list[np.ndarray]]:
    """Per image: array (N, 5) of [class_id, x1, y1, x2, y2] in pixels."""
    from PIL import Image

    gts = []
    for p in images:
        w, h = Image.open(p).size
        lab = labels_dir / f"{p.stem}.txt"
        rows = []
        if lab.exists():
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h])
        gts.append(np.asarray(rows, dtype=float).reshape(-1, 5))
    return gts


def ap50(entries: list[list[list[list[float]]]], gts: list[list[np.ndarray]],
         names: list[str]) -> tuple[float, dict[str, float]]:
    """mAP@0.5 matching Ultralytics' methodology.

    Two details matter for reproducing Ultralytics' number:
      * matching is an optimal one-to-one assignment per image (Hungarian),
        not greedy-by-score -- greedy under-counts true positives and loses
        ~1.6 mAP50 on this dataset;
      * AP is integrated on 101 recall points, not all points.
    """
    from scipy.optimize import linear_sum_assignment

    per_class: dict[str, float] = {}
    for cls in names:
        cid = CLASSES.index(cls)
        slot = NAME_TO_SLOT[cls]

        gt_by_img: dict[int, np.ndarray] = {}
        pred_by_img: dict[int, list[list[float]]] = {}
        n_pos = 0
        for i, gt in enumerate(gts):
            sel = gt[gt[:, 0] == cid] if len(gt) else np.zeros((0, 5))
            gt_by_img[i] = sel
            n_pos += len(sel)
            pred_by_img[i] = entries[i][slot]
        if n_pos == 0:
            continue

        tp_map: dict[int, np.ndarray] = {}
        for i, boxes in pred_by_img.items():
            gt = gt_by_img[i]
            n_p, n_g = len(boxes), len(gt)
            matched = np.zeros(n_p, dtype=bool)
            if n_p and n_g:
                iou_mat = np.zeros((n_p, n_g))
                for a in range(n_p):
                    for b in range(n_g):
                        iou_mat[a, b] = iou_xyxy(np.asarray(boxes[a][:4], dtype=float),
                                                 gt[b, 1:5])
                rows, cols = linear_sum_assignment(-iou_mat)   # maximise IoU
                for a, b in zip(rows, cols):
                    if iou_mat[a, b] >= 0.5:
                        matched[a] = True
            tp_map[i] = matched

        records = sorted(
            ((b[4], i, k) for i, boxes in pred_by_img.items() for k, b in enumerate(boxes)),
            key=lambda t: -t[0],
        )
        if not records:
            per_class[cls] = 0.0
            continue
        tp = np.array([tp_map[i][k] for _, i, k in records], dtype=float)
        fp = 1.0 - tp

        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        recall = ctp / n_pos
        precision = ctp / np.maximum(ctp + cfp, 1e-9)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0.0, 1.0, 101)
        # numpy 2.x renamed trapz -> trapezoid
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        per_class[cls] = float(trapz(np.interp(x, mrec, mpre), x))

    mean = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean, per_class


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--source", default=None, help="image dir; defaults to val for --eval, test2 otherwise")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--imgsz-per-model", default=None,
                    help="comma separated, one per --weights (e.g. 640,640,1024)")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.75)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--eval", action="store_true", help="score against the val labels")
    ap.add_argument("--out", default=None, help="write a submission JSON")
    args = ap.parse_args()

    source = Path(args.source) if args.source else (VAL_IMG if args.eval else TEST_IMG)
    images = sorted(source.glob("*.jpg"))
    if not images:
        print(f"no images in {source}")
        return 1
    print(f"fusing {len(args.weights)} models on {len(images)} images from {source}")
    if args.imgsz_per_model:
        imgsz_list = [int(v) for v in args.imgsz_per_model.split(",")]
        if len(imgsz_list) != len(args.weights):
            print("--imgsz-per-model needs one value per --weights")
            return 1
    else:
        imgsz_list = [args.imgsz] * len(args.weights)
    print(f"imgsz={imgsz_list} conf={args.conf} nms_iou={args.iou} max_det={args.max_det} tta={args.tta}")

    entries = run_models(args.weights, images, imgsz_list, args.conf, args.iou,
                         args.max_det, args.tta, args.batch, args.device)

    if args.eval:
        gts = load_gt(VAL_LBL, images)
        mean, per_class = ap50(entries, gts, CLASSES)
        print()
        print("=" * 60)
        print(f"fused mAP@0.5 = {mean:.4f}   ->  predicted online score {mean * 0.90 * 100:.2f}")
        print("=" * 60)
        for cls, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
            print(f"  {cls:<14}{v:.4f}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.write_text(json.dumps(entries), encoding="utf-8")
        print(f"\nsubmission -> {out}  ({out.stat().st_size / 1024**2:.1f} MiB)")
        per_slot = [sum(len(e[s]) for e in entries) for s in range(len(SUBMISSION_ORDER))]
        print("boxes per slot: " + ", ".join(f"{SUBMISSION_ORDER[s]}={n}" for s, n in enumerate(per_slot)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
