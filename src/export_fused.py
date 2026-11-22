"""Fuse several models and write the official test2 submission.

    python src/export_fused.py --models F,B,H1
    python src/export_fused.py --models F,B,H1 --iou-thr 0.55

Measured on the 489-image leak-free split with the same pycocotools scorer:

    single F 0.7806 | single B 0.7830 | single H1 0.7565
    F+B fused        0.8100   (+2.70 over the best single)
    F+B+H1 fused     0.8215   (+3.84 over the best single)

The +1.15 that H1 adds on top of F+B is the part that matters: H1 alone is the
WEAKEST of the three, yet it still improves the ensemble, because HGNetV2 makes
different mistakes than CSPDarknet.  That is the argument for adding RT-DETR too.

Class mapping goes through class NAMES (src/classes.py::NAME_TO_SLOT), never raw
ids: training order is alphabetical, official slots are not.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import NAME_TO_SLOT, SUBMISSION_ORDER  # noqa: E402
from ensemble_eval import MODELS, wbf_image  # noqa: E402

ROOT = Path(r"E:\X光")
TEST_DIR = ROOT / "data" / "test2"
TEMPLATE = ROOT / "results_test2.json"


def load_fusion_config(path: Path) -> dict:
    """Read the fusion settings produced by ensemble_eval.py.

    Weights and the WBF IoU threshold come from the evaluation that measured them,
    never from constants in this file.  A hard-coded table here previously went
    stale twice: it lacked G and R1, and its default iou_thr (0.55) disagreed with
    the best measured threshold (0.50) for the five-model combination.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["models", "checkpoints", "weights", "iou_thr"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path} is not a fusion config; missing {missing}")
    return data


def predict_chunk(model, names, paths, imgsz, conf, iou, max_det, tta):
    """Return list of {class_name: (boxes Nx4, scores N)} aligned with paths."""
    with contextlib.redirect_stdout(io.StringIO()):
        results = model.predict([str(p) for p in paths], imgsz=imgsz, conf=conf, iou=iou,
                                max_det=max_det, augment=tta, rect=False,
                                verbose=False, device=0)
    out = []
    for res in results:
        per: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        boxes = res.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for name in set(names[int(c)] for c in classes):
                m = np.array([names[int(c)] == name for c in classes])
                per[name] = (xyxy[m], confs[m])
        out.append(per)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="fusion_config.json written by ensemble_eval.py; supplies "
                         "models, per-model weights and the WBF IoU threshold")
    ap.add_argument("--models", default=None, help="override the model list (debugging only)")
    ap.add_argument("--imgsz", type=int, default=None, help="defaults to the config value")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--iou-thr", type=float, default=None,
                    help="defaults to the config value; overriding it means the export no "
                         "longer matches what was evaluated")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--augment", action="store_true", default=None)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results_test2_submit_fused.json"))
    ap.add_argument("--dry-run", type=int, default=0)
    args = ap.parse_args()

    cfg = load_fusion_config(Path(args.config))
    imgsz_map_cfg = cfg.get("imgsz_map") or {}
    keys = [k.strip() for k in (args.models.split(",") if args.models else cfg["models"])]
    weights = cfg["weights"]
    iou_thr = args.iou_thr if args.iou_thr is not None else float(cfg["iou_thr"])
    imgsz = args.imgsz if args.imgsz is not None else int(cfg["imgsz"])
    conf = args.conf if args.conf is not None else float(cfg["conf"])
    tta = args.augment if args.augment is not None else bool(cfg["tta"])

    if args.iou_thr is not None and abs(float(args.iou_thr) - float(cfg["iou_thr"])) > 1e-9:
        print(f"WARNING: WBF iou_thr {args.iou_thr} differs from the evaluated "
              f"{cfg['iou_thr']}; this export is not the configuration that was measured")

    missing = [k for k in keys if k not in cfg["checkpoints"]]
    if missing:
        print(f"!! {args.config} has no checkpoint for {missing}")
        return 1
    for k in keys:
        if k not in weights:
            print(f"!! {args.config} has no weight for {k}")
            return 1
        if not Path(cfg["checkpoints"][k]).is_file():
            print(f"missing checkpoint for {k}: {cfg['checkpoints'][k]}")
            return 1

    images = sorted(TEST_DIR.glob("*.jpg"))
    if not images:
        print(f"no images in {TEST_DIR}")
        return 1
    if TEMPLATE.exists():
        expected = len(json.loads(TEMPLATE.read_text(encoding="utf-8")))
        if expected != len(images):
            print(f"!! template has {expected} entries but {len(images)} test images")
            return 1
    if args.dry_run:
        images = images[: args.dry_run]

    from ultralytics import YOLO
    from PIL import Image

    models = []
    for k in keys:
        m = YOLO(str(cfg["checkpoints"][k]))
        names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
        bad = [n for n in names if n not in NAME_TO_SLOT]
        if bad:
            print(f"!! {k}: classes outside the submission order: {bad}")
            return 1
        models.append((k, m, names))
        print(f"  loaded {k}: {names}")

    print(f"config {cfg['run_id']}  expected mAP50 {cfg.get('expected_mAP50')}")
    print(f"images {len(images)}  models {keys}  WBF iou {iou_thr}  imgsz {imgsz}  tta {tta}")
    entries, total, skipped = [], 0, 0

    for i in range(0, len(images), args.batch):
        chunk = images[i: i + args.batch]
        per_model = [predict_chunk(m, names, chunk, int(imgsz_map_cfg.get(k, imgsz)),
                                    conf, args.iou, args.max_det, tta)
                     for (k, m, names) in models]
        for j, path in enumerate(chunk):
            with Image.open(path) as im:
                w, h = im.size
            entry: list[list[list[float]]] = [[] for _ in range(len(SUBMISSION_ORDER))]
            for cls_name in NAME_TO_SLOT:
                bb, ss = [], []
                for mi, (_k, _m, _n) in enumerate(models):
                    boxes, scores = per_model[mi][j].get(cls_name, (np.zeros((0, 4)), np.zeros(0)))
                    bb.append(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
                    ss.append(np.asarray(scores, dtype=np.float32))
                fb, fs = wbf_image(bb, ss, [weights[k] for k in keys], iou_thr)
                slot = NAME_TO_SLOT[cls_name]
                eps = 1e-3
                for (x1, y1, x2, y2), sc in zip(fb, fs):
                    x1 = round(min(max(float(x1), eps), float(w)), 3)
                    y1 = round(min(max(float(y1), eps), float(h)), 3)
                    x2 = round(min(max(float(x2), eps), float(w)), 3)
                    y2 = round(min(max(float(y2), eps), float(h)), 3)
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        skipped += 1
                        continue
                    entry[slot].append([x1, y1, x2, y2, round(float(sc), 6)])
                    total += 1
            entries.append(entry)
        print(f"  {min(i + args.batch, len(images))}/{len(images)} images, {total} boxes", flush=True)

    out = Path(args.out)
    if args.dry_run:
        out = out.with_suffix(".dryrun.json")
    out.write_text(json.dumps(entries), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"entries {len(entries)}, boxes {total}, dropped degenerate {skipped}")
    per_slot = [sum(len(e[s]) for e in entries) for s in range(len(SUBMISSION_ORDER))]
    for slot, count in enumerate(per_slot):
        print(f"  slot {slot} ({SUBMISSION_ORDER[slot]:<13}): {count:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
