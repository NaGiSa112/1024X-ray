"""Evaluate a trained run on the fixed validation split and log the result.

    python src/evaluate.py --weights runs/first_test/A_coco640/weights/best.pt --tag A

Prints overall mAP / precision / recall plus per-class AP50 for all eight
classes, and appends one row to `runs/training_log.md` so the experiment table
for the report fills itself in.

`--conf 0.001` matches the plan: evaluation must use a low threshold so the full
PR curve is covered.  The `conf=0.25` used for eyeballing pictures is not an
evaluation threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "training_log.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True, help="short experiment label, e.g. A")
    ap.add_argument("--data", default=str(ROOT / "data" / "yolo" / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--cm-conf", type=float, default=0.25,
                    help="confidence used for the confusion matrix (a usable threshold)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--augment", action="store_true", help="enable test-time augmentation")
    ap.add_argument("--half", action="store_true", help="use FP16 evaluation where supported")
    ap.add_argument("--square", action="store_true",
                    help="disable rectangular batches to match mixed-shape batch prediction")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-cm", action="store_true",
                    help="skip the second validation pass used for the confusion matrix")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (Path.cwd() / weights).resolve()
    if not weights.exists():
        print(f"weights not found: {weights}")
        return 1

    eval_config = {
        "data": str(Path(args.data).expanduser().resolve()),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "cm_conf": args.cm_conf,
        "iou": 0.7,
        "max_det": 300,
        "augment": args.augment,
        "half": args.half,
        "rect": not args.square,
        "workers": args.workers,
        "skip_cm": args.skip_cm,
    }
    val_kwargs = {k: v for k, v in eval_config.items() if k not in {"cm_conf", "skip_cm"}}
    val_kwargs.update(
        device=args.device,
        plots=True,
        project=str(ROOT / "runs" / "first_test"),
        exist_ok=True,
        verbose=False,
    )
    model = YOLO(str(weights))
    metrics = model.val(**val_kwargs, name=f"{args.tag}_eval")

    box = metrics.box
    names = model.names
    print()
    print("=" * 66)
    print(f"Experiment {args.tag}  |  {weights}")
    print("=" * 66)
    print(f"  mAP50      : {box.map50:.4f}")
    print(f"  mAP50-95   : {box.map:.4f}")
    print(f"  precision  : {box.mp:.4f}")
    print(f"  recall     : {box.mr:.4f}")
    print()
    print(f"  {'class':<14}{'AP50':>8}{'AP50-95':>10}")
    per_class: dict[str, dict[str, float]] = {}
    # box.ap50 / box.ap are indexed by position within box.ap_class_index
    for pos, class_idx in enumerate(box.ap_class_index):
        name = names[int(class_idx)]
        ap50 = float(box.ap50[pos])
        ap5095 = float(box.ap[pos].mean())
        per_class[name] = {"ap50": round(ap50, 4), "ap50_95": round(ap5095, 4)}
        print(f"  {name:<14}{ap50:>8.4f}{ap5095:>10.4f}")
    for name in CLASSES:
        if name not in per_class:
            per_class[name] = {"ap50": 0.0, "ap50_95": 0.0}
            print(f"  {name:<14}{0.0:>8.4f}{0.0:>10.4f}   (absent from val)")

    rare = ("tool", "laptop", "knife")
    print()
    print("  rare classes:", ", ".join(f"{n} {per_class[n]['ap50']:.3f}" for n in rare))

    # ---- raw confusion matrix, computed at a *usable* threshold ---------
    # The main val runs at conf=0.001 so the whole PR curve is covered, which
    # means "every class is predicted everywhere" (11734 glassbottle predictions
    # for 881 true boxes).  A matrix built from that is not interpretable, so the
    # matrix is taken from a second pass at --cm-conf.
    #
    # Ultralytics accumulates matrix[detection_class, gt_class], so a column holds
    # every box whose ground truth is that class: diagonal/column = recall,
    # diagonal/row = precision.  Saved as CSV so the report quotes exact counts.
    cm_csv = weights.parent.parent / f"confusion_matrix_{args.tag}.csv"
    cm_table = None
    if args.skip_cm:
        print("  confusion matrix pass skipped (--skip-cm)")
    else:
        try:
            import csv as _csv

            cm_metrics = model.val(
                **{**val_kwargs, "conf": args.cm_conf},
                name=f"{args.tag}_cm",
            )
            cm = cm_metrics.confusion_matrix.matrix
            labels = [str(names[i]) for i in range(len(names))] + ["background"]
            with cm_csv.open("w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.writer(fh)
                w.writerow(["predicted\\true"] + labels)
                for i, row in enumerate(cm):
                    w.writerow([labels[i]] + [int(v) for v in row])

            print()
            print(f"  confusion matrix at conf={args.cm_conf}")
            print(f"  {'class':<14}{'pred':>6}{'true':>6}{'recall':>8}{'precision':>10}")
            cm_table = {}
            for i, name in enumerate(labels[:-1]):
                true_n = float(cm[:, i].sum())
                pred_n = float(cm[i, :].sum())
                diag = float(cm[i, i])
                rec = diag / true_n if true_n else 0.0
                prec = diag / pred_n if pred_n else 0.0
                cm_table[name] = {"pred": int(pred_n), "true": int(true_n),
                                  "recall": round(rec, 4), "precision": round(prec, 4)}
                print(f"  {name:<14}{int(pred_n):>6}{int(true_n):>6}{rec:>8.3f}{prec:>10.3f}")
            missed = {n: round(1.0 - cm_table[n]["recall"], 3) for n in cm_table}
            worst = sorted(missed.items(), key=lambda kv: -kv[1])[:3]
            print("  most missed (recall gap): "
                  + ", ".join(f"{n} {v:.2f}" for n, v in worst))
            print(f"\n  confusion matrix -> {cm_csv}")
        except Exception as exc:  # noqa: BLE001 - never fail the run over a plot
            print(f"  (confusion matrix not saved: {type(exc).__name__}: {exc})")

    record = {
        "tag": args.tag,
        "weights": str(weights),
        "time": datetime.now().isoformat(timespec="seconds"),
        "eval_config": eval_config,
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "confusion": cm_table,
        "notes": args.notes,
    }
    json_path = weights.parent.parent / f"metrics_{args.tag}.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  metrics -> {json_path}")

    # ---- append to the shared experiment table -------------------------
    row = (
        f"| {args.tag} | `{weights.parent.parent.relative_to(ROOT).as_posix()}` "        f"| {record['mAP50']:.4f} | {record['mAP50_95']:.4f} "
        f"| {record['precision']:.4f} | {record['recall']:.4f} "
        f"| {per_class['tool']['ap50']:.4f} | {per_class['knife']['ap50']:.4f} "
        f"| {per_class['laptop']['ap50']:.4f} | {args.notes} |"
    )
    header = (
        "# 实验结果记录\n\n"
        "| 实验 | 产物目录 | mAP50 | mAP50-95 | Precision | Recall "
        "| tool AP50 | knife AP50 | laptop AP50 | 备注 |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    with LOG.open("a", encoding="utf-8") as fh:
        if LOG.stat().st_size == 0:
            fh.write(header)
        fh.write(row + "\n")
    print(f"  logged -> {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
