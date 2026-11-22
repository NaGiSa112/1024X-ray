"""Train YOLO while retaining the checkpoint with the highest validation AP50.

This keeps Ultralytics' normal best.pt and fitness unchanged.  The additional
weights/best_map50.pt is copied only after the completed epoch's last.pt has
been saved.  --dry-run validates paths and prints configuration without
importing Ultralytics or torch, creating directories, or starting training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC = "metrics/mAP50(B)"


def _atomic_copy(source: Path, destination: Path) -> None:
    """Replace a checkpoint only after its complete copy is available."""
    handle, name = tempfile.mkstemp(prefix=".best_map50-", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, destination: Path) -> None:
    handle, name = tempfile.mkstemp(prefix=".best_map50-", suffix=".json.tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_best_map50(trainer) -> None:
    """Top-level callback, invoked after validation and saving last.pt."""
    if METRIC not in trainer.metrics:
        raise RuntimeError(f"AP50 checkpoint selection requires validation metric {METRIC!r}")
    value = float(trainer.metrics[METRIC])
    if not math.isfinite(value):
        print("AP50 is non-finite; retaining the previous best_map50.pt", flush=True)
        return
    if value <= getattr(trainer, "_best_map50_value", -math.inf):
        return
    source = Path(trainer.last)
    if not source.is_file():
        raise FileNotFoundError(f"Completed epoch checkpoint is missing: {source}")
    destination = source.parent / "best_map50.pt"
    record = {
        "metric": METRIC,
        "value": value,
        "epoch": int(trainer.epoch) + 1,
        "epoch_numbering": "one-based",
        "weights": str(destination.resolve()),
        "data": str(trainer.args.data),
        "selection": "maximum validation AP50; ties retain the earlier epoch",
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_copy(source, destination)
    _atomic_json(record, source.parent / "best_map50.json")
    trainer._best_map50_value = value
    print(f"AP50 best: epoch {record['epoch']}, {value:.6f} -> {destination}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(ROOT / "weights" / "yolov8_eds1.pt"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True, help="new run directory name, without path separators")
    parser.add_argument("--project", default=str(ROOT / "runs" / "first_test"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--lr0", type=float, default=0.0008)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--cls", type=float, default=0.5,
                        help="classification loss gain; the leaderboard metric is AP50, which "
                             "only rewards ranking, so a slightly higher value can help "
                             "(the 2021 winner of this competition used +25%% on this term)")
    parser.add_argument("--box", type=float, default=7.5, help="box loss gain")
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its last.pt "
                             "(keeps optimizer/EMA/scaler state and the LR schedule)")
    parser.add_argument("--map50-floor", type=float, default=None,
                        help="AP50 already reached before the interruption. A resumed "
                             "run restarts the internal best-tracker from scratch, which "
                             "would let a WORSE epoch overwrite best_map50.pt; seeding the "
                             "floor with the pre-crash best prevents that.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def training_config(args: argparse.Namespace) -> dict:
    for name in ("epochs", "imgsz", "batch"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not math.isfinite(args.lr0) or args.lr0 <= 0:
        raise ValueError("--lr0 must be finite and positive")
    if args.workers < 0 or not 0 <= args.close_mosaic <= args.epochs:
        raise ValueError("workers must be nonnegative; close-mosaic must be between zero and epochs")
    if not math.isfinite(args.warmup_epochs) or args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be finite and nonnegative")
    for name in ("scale", "mosaic", "mixup", "flipud"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name} must be between zero and one")
    for name in ("cls", "box"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be finite and positive")
    if not args.name or args.name in {".", ".."} or any(c in args.name for c in '/\\:'):
        raise ValueError("--name must be a directory name without path separators or a drive prefix")
    weights = Path(args.weights).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data}")
    project = Path(args.project).expanduser().resolve()
    run = project / args.name
    # A resume intentionally continues in an existing run directory, so the
    # "must be empty" guard only applies to fresh runs.
    if not args.resume and run.exists() and (not run.is_dir() or any(run.iterdir())):
        raise FileExistsError(f"Run path already exists and is not an empty directory: {run}. Choose a new --name.")
    if args.resume and not weights.is_file():
        raise FileNotFoundError(f"--resume needs the checkpoint to resume from: {weights}")
    return {
        "model": str(weights),
        "data": str(data),
        "resume": bool(args.resume),
        "project": str(project),
        "name": args.name,
        "exist_ok": True,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": args.warmup_epochs,
        "patience": 0,
        "amp": True,
        "cache": False,
        "plots": True,
        "val": True,
        "save": True,
        "save_period": args.save_period,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.15,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": args.scale,
        "fliplr": 0.5,
        "flipud": args.flipud,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": 0.0,
        "close_mosaic": args.close_mosaic,
        "cls": args.cls,
        "box": args.box,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = training_config(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    # Keep heavyweight imports below the dry-run branch and main guard for
    # Windows worker spawning and configuration-only inspection.
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import strip_optimizer

    model = YOLO(config.pop("model"))
    model.add_callback("on_model_save", save_best_map50)

    if args.map50_floor is not None:
        floor = float(args.map50_floor)

        def seed_floor(trainer) -> None:
            # Runs before the first epoch; save_best_map50 only replaces the
            # checkpoint when a later epoch beats this value.
            trainer._best_map50_value = floor
            print(f"best_map50 floor seeded at {floor:.6f} (pre-interruption best)",
                  flush=True)

        model.add_callback("on_train_start", seed_floor)

    model.train(**config)
    best_map50 = Path(model.trainer.last).parent / "best_map50.pt"
    if not best_map50.is_file():
        raise RuntimeError("Training finished without best_map50.pt; inspect validation and checkpoint logs")
    # The epoch/score remains in best_map50.json after optimizer stripping.
    strip_optimizer(best_map50)
    print(f"Finished. AP50-selected checkpoint: {best_map50}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
