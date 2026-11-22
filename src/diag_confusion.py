"""Build a real confusion matrix for one checkpoint on the 489-image split.

    python src/diag_confusion.py --weights runs/first_test/S3_semi_round2/weights/best_map50.pt
    python src/diag_confusion.py --weights ... --imgsz 1280 --conf 0.25

evaluate.py has a --skip-cm flag, but the CSV it writes when the flag is absent is
0 bytes -- every confusion_matrix_*.csv in runs/ is empty, so any conclusion drawn
from that file was drawn from nothing.  This script computes the matrix directly:
predictions are matched to ground truth greedily by descending score at IoU >= 0.5,
and each ground-truth box lands in exactly one cell:

    predicted class == true class   -> diagonal (correct)
    predicted class != true class   -> off-diagonal (confusion)
    no prediction matched           -> background column (miss)

Unmatched predictions are collected in the background row, which is how many
false positives each class produces.  This is the information needed to tell
"the model cannot see this class" apart from "the model sees it but calls it
something else", and those two need opposite fixes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CLASSES = ["drinkbottle", "glassbottle", "knife", "laptop",
           "metalbottle", "pressuretank", "tool", "umbrella"]
ROOT = Path(r"E:\X光")
VAL_IMG = ROOT / "data" / "yolo_clean" / "images" / "val"
VAL_LBL = ROOT / "data" / "yolo_clean" / "labels" / "val"
MANIFEST = ROOT / "runs" / "audit" / "E_content_disjoint_val.txt"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.where((aa[:, None] + bb[None, :] - inter) > 0,
                    inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    stems = []
    if MANIFEST.is_file():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            tail = line.strip().replace("\\", "/").rsplit("/", 1)[-1]
            if tail.lower().endswith(".jpg"):
                stems.append(tail[:-4])
    wanted = set(stems)
    imgs = [p for p in sorted(VAL_IMG.glob("*.jpg")) if not wanted or p.stem in wanted]
    print(f"val images {len(imgs)}  weights {Path(args.weights).name}  imgsz {args.imgsz}  conf {args.conf}")

    model = YOLO(args.weights)
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)

    n = len(CLASSES)
    conf_mat = np.zeros((n + 1, n + 1), dtype=int)   # rows: truth (+background), cols: pred
    tp = np.zeros(n, dtype=int)
    fn = np.zeros(n, dtype=int)
    fp_bg = np.zeros(n, dtype=int)

    for start in range(0, len(imgs), args.batch):
        chunk = imgs[start: start + args.batch]
        with contextlib.redirect_stdout(io.StringIO()):
            results = model.predict([str(p) for p in chunk], imgsz=args.imgsz, conf=args.conf,
                                    iou=0.7, max_det=300, rect=False, verbose=False, device=0)
        for path, res in zip(chunk, results):
            with Image.open(path) as im:
                w, h = im.size
            gt = []
            lbl = VAL_LBL / f"{path.stem}.txt"
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    p = line.split()
                    c = CLASSES.index(names[int(float(p[0]))]) if False else int(float(p[0]))
                    cx, cy, bw, bh = map(float, p[1:5])
                    gt.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                   (cx + bw / 2) * w, (cy + bh / 2) * h]))
            preds = []
            if res.boxes is not None and len(res.boxes):
                b = res.boxes.xyxy.cpu().numpy()
                s = res.boxes.conf.cpu().numpy()
                cc = res.boxes.cls.cpu().numpy().astype(int)
                for box, sc, cl in zip(b, s, cc):
                    preds.append((CLASSES.index(names[int(cl)]), float(sc), box))
            preds.sort(key=lambda t: -t[1])

            used_pred = np.zeros(len(preds), dtype=bool)
            used_gt = np.zeros(len(gt), dtype=bool)
            if gt and preds:
                g = np.asarray([t[1] for t in gt], dtype=np.float32)
                p = np.asarray([t[2] for t in preds], dtype=np.float32)
                ious = iou_matrix(g, p)
                for gi in range(len(gt)):
                    best, best_j = 0.0, -1
                    for pi in range(len(preds)):
                        if used_pred[pi]:
                            continue
                        if ious[gi, pi] > best:
                            best, best_j = float(ious[gi, pi]), pi
                    if best_j >= 0 and best >= args.iou_thr:
                        used_pred[best_j] = True
                        used_gt[gi] = True
                        conf_mat[gt[gi][0], preds[best_j][0]] += 1
                        if gt[gi][0] == preds[best_j][0]:
                            tp[gt[gi][0]] += 1
            for gi in range(len(gt)):
                if not used_gt[gi]:
                    fn[gt[gi][0]] += 1
                    conf_mat[gt[gi][0], n] += 1          # truth -> background
            for pi in range(len(preds)):
                if not used_pred[pi]:
                    fp_bg[preds[pi][0]] += 1
                    conf_mat[n, preds[pi][0]] += 1       # background -> predicted

        print(f"  {min(start + args.batch, len(imgs))}/{len(imgs)}", flush=True)

    print("\n=== per class (conf {:.2f}, IoU {:.2f}) ===".format(args.conf, args.iou_thr))
    print(f"{'class':<14}{'GT':>6}{'TP':>6}{'FN':>6}{'recall':>8}{'FP(bg)':>8}{'prec':>8}")
    for i, cls in enumerate(CLASSES):
        tot = tp[i] + fn[i]
        rec = tp[i] / tot if tot else float("nan")
        prec = tp[i] / (tp[i] + fp_bg[i]) if (tp[i] + fp_bg[i]) else float("nan")
        print(f"{cls:<14}{tot:>6}{tp[i]:>6}{fn[i]:>6}{rec:>8.3f}{fp_bg[i]:>8}{prec:>8.3f}")

    print("\n=== where each class GOES when mis-classified (row = truth) ===")
    print(f"{'truth':<14}" + "".join(f"{c[:9]:>11}" for c in CLASSES) + f"{'missed':>9}")
    for i, cls in enumerate(CLASSES):
        row = "".join(f"{conf_mat[i, j]:>11}" for j in range(n))
        print(f"{cls:<14}{row}{conf_mat[i, n]:>9}")

    print("\n=== biggest confusions (truth -> predicted) ===")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and conf_mat[i, j] > 0:
                pairs.append((int(conf_mat[i, j]), CLASSES[i], CLASSES[j]))
    for cnt, a, b in sorted(pairs, reverse=True)[:8]:
        print(f"  {a:<14} -> {b:<14} {cnt}")

    out = ROOT / "runs" / "audit" / f"confusion_{Path(args.weights).parent.parent.name}_{args.imgsz}.json"
    import json
    out.write_text(json.dumps({
        "weights": str(args.weights), "imgsz": args.imgsz, "conf": args.conf,
        "iou_thr": args.iou_thr, "n_images": len(imgs), "classes": CLASSES,
        "matrix_rows_truth_cols_pred_plus_background": conf_mat.tolist(),
        "per_class": {c: {"gt": int(tp[i] + fn[i]), "tp": int(tp[i]), "fn": int(fn[i]),
                          "fp_background": int(fp_bg[i])} for i, c in enumerate(CLASSES)},
    }, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
