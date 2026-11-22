"""Search WBF scoring policies using cached per-model predictions.

    python src/fuse_policy_search.py --run-dir runs/audit/ensemble/20261122_045206_F-B-H1-D-G-S1-S2

Why: the fusion gains so far all came from adding members, while the fusion RULE
itself was never questioned.  The current rule multiplies a cluster score by
n_voters / n_models, which with seven members means a box found by three models is
scaled to 3/7 of its score.  AP50 only cares about ranking, and the ranking WITHIN a
class is exactly what that factor changes, so it is worth measuring instead of
assuming.

Cached `preds_single_*.json` from an existing run are reused, so each policy costs
about a minute instead of a 40-minute re-inference.  Policies differ only in how a
cluster's score is computed; box averaging is identical across all of them.

Overfitting guard: a handful of hand-specified policies is compared, not a free
search over many parameters, and any improvement smaller than the spread between
policies is treated as noise rather than adopted.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CLASSES = ["drinkbottle", "glassbottle", "knife", "laptop",
           "metalbottle", "pressuretank", "tool", "umbrella"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def wbf(boxes_by_model, scores_by_model, weights, iou_thr, policy):
    allb, alls, allm = [], [], []
    for mi, (b, s) in enumerate(zip(boxes_by_model, scores_by_model)):
        for k in range(len(b)):
            allb.append(b[k]); alls.append(s[k]); allm.append(mi)
    if not allb:
        return np.zeros((0, 4)), np.zeros(0)
    allb = np.asarray(allb, dtype=np.float32)
    alls = np.asarray(alls, dtype=np.float32)
    allm = np.asarray(allm, dtype=np.int64)
    order = np.argsort(-alls)
    allb, alls, allm = allb[order], alls[order], allm[order]

    n = len(boxes_by_model)
    used = np.zeros(len(alls), bool)
    fb, fs = [], []
    for i in range(len(alls)):
        if used[i]:
            continue
        ious = iou_matrix(allb[i: i + 1], allb)[0]
        members = np.where((ious >= iou_thr) & (~used))[0]
        seen, keep = set(), []
        for j in members:
            if allm[j] in seen:
                continue
            seen.add(allm[j]); keep.append(j)
        keep = np.asarray(keep, dtype=np.int64)
        if keep.size == 0:
            continue
        used[keep] = True
        w = np.array([weights[allm[j]] for j in keep], dtype=np.float32)
        sc = alls[keep]
        fb.append((allb[keep] * (sc * w)[:, None]).sum(0) / np.maximum((sc * w).sum(), 1e-9))
        base = float(sc.mean())
        frac = min(len(seen), n) / n
        if policy == "vote":              # current production rule
            fs.append(base * frac)
        elif policy == "vote_sqrt":
            fs.append(base * np.sqrt(frac))
        elif policy == "no_penalty":
            fs.append(base)
        elif policy == "vote_sq":
            fs.append(base * frac * frac)
        elif policy == "sum_norm":        # sum of weighted scores / total weight
            fs.append(float((sc * w).sum() / max(w.sum(), 1e-9)))
        else:
            raise ValueError(policy)
    return np.asarray(fb, dtype=np.float32), np.asarray(fs, dtype=np.float32)


def score(gt_path: Path, preds: list[dict], label: str) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    tmp = Path(gt_path).parent / f"policy_{label}.json"
    tmp.write_text(json.dumps(preds), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(str(tmp))
        ev = COCOeval(gt, dt, "bbox")
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate(); ev.accumulate()
    prec = ev.eval["precision"]
    per = {}
    for k, cls in enumerate(CLASSES):
        arr = prec[0, :, k, 0, 2]
        arr = arr[arr > -1]
        per[cls] = float(np.mean(arr)) if arr.size else float("nan")
    return {"label": label, "mAP50": float(np.nanmean(list(per.values()))), "per_class": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iou-thr", default="0.5,0.55,0.6")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "fusion_config.json").read_text(encoding="utf-8"))
    keys = cfg["models"]
    weights = cfg["weights"]
    gt_path = run_dir / "gt.json"
    if not gt_path.is_file():
        print(f"missing gt: {gt_path}")
        return 1

    # invert image_id -> index and collect predictions per image per class
    per_model: dict[str, dict[int, list[tuple[str, float, list[float]]]]] = {}
    n_images = 0
    for k in keys:
        p = run_dir / f"preds_single_{k}.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        by_img: dict[int, list[tuple[str, float, list[float]]]] = defaultdict(list)
        for r in data:
            x, y, w, h = r["bbox"]
            by_img[int(r["image_id"])].append((r["category_id"], float(r["score"]),
                                               [x, y, x + w, y + h]))
        per_model[k] = by_img
        n_images = max(n_images, max(by_img) if by_img else 0)
    print(f"images {n_images}  models {keys}")

    results = []
    for policy in ["vote", "vote_sqrt", "sum_norm", "vote_sq", "no_penalty"]:
        for iou_thr in [float(x) for x in args.iou_thr.split(",")]:
            fused = []
            for img in range(1, n_images + 1):
                for cls in CLASSES:
                    bb, ss = [], []
                    for k in keys:
                        arr = [t for t in per_model[k].get(img, []) if t[0] == cls]
                        bb.append(np.asarray([t[2] for t in arr], dtype=np.float32).reshape(-1, 4))
                        ss.append(np.asarray([t[1] for t in arr], dtype=np.float32))
                    fb, fs = wbf(bb, ss, [weights[k] for k in keys], iou_thr, policy)
                    for box, sc in zip(fb, fs):
                        fused.append({"image_id": img, "category_id": cls,
                                      "bbox": [float(box[0]), float(box[1]),
                                               float(box[2] - box[0]), float(box[3] - box[1])],
                                      "score": float(sc)})
            res = score(gt_path, fused, f"{policy}_{int(iou_thr*100)}")
            res["policy"] = policy
            res["iou_thr"] = iou_thr
            res["n_boxes"] = len(fused)
            results.append(res)
            print(f"  {policy:<12} iou {iou_thr}  boxes {len(fused):>6}  mAP50 {res['mAP50']:.4f}")

    results.sort(key=lambda r: -r["mAP50"])
    print("\n=== ranked ===")
    for r in results[:5]:
        print(f"  {r['policy']:<12} iou {r['iou_thr']}  {r['mAP50']:.4f}")
    best = results[0]
    print(f"\nbest: {best['policy']} @ iou {best['iou_thr']} -> {best['mAP50']:.4f}")
    print("per class:")
    for c in CLASSES:
        print(f"  {c:<14}{best['per_class'][c]:.4f}")
    (run_dir / "policy_search.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n-> {run_dir / 'policy_search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
