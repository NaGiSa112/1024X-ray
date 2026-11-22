"""Train RT-DETR with the same AP50-best-checkpoint trick used for YOLO.

    python src/train_rtdetr.py --data data/yolo_night75b/data.yaml --name R1_rtdetr1024

Why RT-DETR: every submission so far has been a YOLOv8, so an ensemble of those
models shares almost all of its errors.  RT-DETR is a transformer detector, and
the jgen/xray-comparative-evaluation weights give it X-ray domain pretraining --
the same source as our YOLOv8 EDS-1 initialisation, whose domain pretraining was
worth +4.5 mAP50.  The HiXray checkpoint has 8 classes, just like this task.

Caveats recorded on purpose:
  * the HiXray class head has different names (PO1, Tablet, Water, ...), so the
    8-row classification head starts from the wrong semantics and has to relearn.
    The backbone/neck/decoder weights are the part worth keeping.
  * ultralytics' RT-DETR dataloader does not apply mosaic/mixup, so those two
    arguments are accepted but have no effect.

`best_map50.pt` is saved by reusing save_best_map50 from train_map50.py, because
Ultralytics' own "best" is chosen on fitness = 0.1*mAP50 + 0.9*mAP50-95 and the
leaderboard only scores AP50.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=str(ROOT / "weights" / "rtdetr_hixray.pt"))
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default=str(ROOT / "runs" / "first_test"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr0", type=float, default=0.0008)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0, help="training-set fraction, for smoke tests")
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--deterministic", type=int, default=0,
                   help="1 forces deterministic kernels. Off by default: RT-DETR's "
                        "grid_sampler_2d_backward has no deterministic CUDA "
                        "implementation, so enabling it only produces warnings "
                        "(and, through PowerShell's stderr handling, can kill the run)")
    p.add_argument("--dry-run", action="store_true")
    return p


def config_of(args: argparse.Namespace) -> dict:
    for name in ("epochs", "imgsz", "batch"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not (0 < args.fraction <= 1):
        raise ValueError("--fraction must be in (0, 1]")
    if not math.isfinite(args.lr0) or args.lr0 <= 0:
        raise ValueError("--lr0 must be finite and positive")
    weights = Path(args.weights).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data}")
    return {
        "model": str(weights),
        "data": str(data),
        "project": str(Path(args.project).expanduser().resolve()),
        "name": args.name,
        "exist_ok": True,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "patience": 0,
        "amp": True,
        "cache": False,
        "plots": True,
        "val": True,
        "save": True,
        "save_period": 10,
        "fraction": args.fraction,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.15,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.3,
        "fliplr": 0.5,
        "flipud": 0.5,
        "mosaic": 0.0,      # RT-DETR dataloader ignores mosaic; 0 keeps intent honest
        "mixup": 0.0,
        "close_mosaic": min(args.close_mosaic, args.epochs - 1) if args.epochs > 1 else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_of(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import strip_optimizer

    from train_map50 import save_best_map50

    model = RTDETR(config.pop("model"))
    print(f"loaded RT-DETR: {len(model.names)} classes, "
          f"{sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M params", flush=True)
    model.add_callback("on_model_save", save_best_map50)
    model.train(**config)

    best = Path(model.trainer.last).parent / "best_map50.pt"
    if best.is_file():
        strip_optimizer(best)
        print(f"Finished. AP50-selected checkpoint: {best}", flush=True)
    else:
        print("Finished without best_map50.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
