"""Build the experiment comparison table from per-run metrics JSON files.

    python src/compare_runs.py

Reads every `runs/first_test/*/metrics_<tag>.json` written by `src/evaluate.py`,
prints one table across experiments and writes `runs/comparison.md`, which is the
table the report can quote directly.

Experiments are ordered A, B, C when present; anything else is appended
alphabetically.  Per-class AP50 is reported for all eight classes even when a run
did not evaluate a class, so columns line up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "first_test"
OUT = ROOT / "runs" / "comparison.md"

# model size differs between experiments, so record it next to the numbers
MODEL_NOTE = {
    "A": "yolov8m (25.90M), COCO init",
    "B_eds640": "yolov8l (43.64M), EDS-1 init",
    "C_eds_aug640": "yolov8l (43.64M), EDS-1 init, stronger aug",
    "D_ft1024": "high-res fine-tune at 1024",
}


def evaluation_display(record: dict) -> tuple[str, str, str, str]:
    """Describe recorded inference settings without guessing unknown runs."""
    config = record.get("eval_config") or {}
    legacy = record.get("tag") in {"A", "B_eds640", "C_eds_aug640"}
    imgsz = config.get("imgsz")
    if imgsz is None:
        size_text = "640（历史记录）" if legacy else "未记录"
    elif isinstance(imgsz, (list, tuple)):
        size_text = "×".join(str(v) for v in imgsz)
    else:
        size_text = str(imgsz)
    augment = config.get("augment")
    if augment is None:
        tta_text = "关（历史记录）" if legacy else "未记录"
    else:
        tta_text = "开" if augment else "关"
    conf_text = str(config["conf"]) if "conf" in config else "未记录"
    rect = config.get("rect")
    if rect is None:
        padding_text = "未记录"
    else:
        padding_text = "矩形（rect=True）" if rect else "方形（rect=False）"
    return size_text, tta_text, conf_text, padding_text


def main() -> int:
    records = []
    for path in sorted(RUNS.glob("*/metrics_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["_path"] = path
        records.append(data)

    if not records:
        print(f"no metrics_*.json under {RUNS}")
        return 1

    order = {"A": 0, "B_eds640": 1, "C_eds_aug640": 2, "D_ft1024": 3}
    records.sort(key=lambda r: (order.get(r["tag"], 99), r["tag"]))
    seen = {r["tag"] for r in records}

    lines: list[str] = []
    lines.append("# 实验评估对照：" + " / ".join(r["tag"] for r in records))
    lines.append("")
    lines.append("由 `src/compare_runs.py` 根据各实验 metrics JSON 汇总。评估尺寸、TTA、阈值和填充方式"
                 "按各条记录列出；不同推理设置的结果应分别比较。A/B/C 的旧记录未保存"
                 " eval_config，已知历史设置标注为 640、TTA 关；其他缺失设置不作推断。")
    lines.append("")
    lines.append("| 实验 | 模型 / 初始化 | 评估尺寸 | TTA | conf | 填充方式 | mAP50 | mAP50-95 | P | R |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for r in records:
        imgsz, tta, conf, padding = evaluation_display(r)
        lines.append(
            f"| {r['tag']} | {MODEL_NOTE.get(r['tag'], '')} | {imgsz} | {tta} | {conf} | {padding} | {r['mAP50']:.4f} | "
            f"{r['mAP50_95']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )

    lines.append("")
    lines.append("## 各类 AP50")
    lines.append("")
    header = "| 类别 | " + " | ".join(r["tag"] for r in records) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---:" for _ in records) + " |")
    for cls in CLASSES:
        cells = []
        for r in records:
            v = (r.get("per_class") or {}).get(cls)
            cells.append(f"{v['ap50']:.4f}" if v else "-")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 各类 AP50-95")
    lines.append("")
    lines.append(header)
    lines.append("| --- | " + " | ".join("---:" for _ in records) + " |")
    for cls in CLASSES:
        cells = []
        for r in records:
            v = (r.get("per_class") or {}).get(cls)
            cells.append(f"{v['ap50_95']:.4f}" if v else "-")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")

    # deltas, only when both sides exist
    if "A" in seen:
        base = next(r for r in records if r["tag"] == "A")
        lines.append("")
        lines.append("## 相对 A 的变化（mAP50 百分点）")
        lines.append("")
        lines.append("| 实验 | ΔmAP50 | ΔmAP50-95 |")
        lines.append("| --- | ---: | ---: |")
        for r in records:
            if r["tag"] == "A":
                continue
            d50 = (r["mAP50"] - base["mAP50"]) * 100
            d = (r["mAP50_95"] - base["mAP50_95"]) * 100
            lines.append(f"| {r['tag']} | {d50:+.2f} | {d:+.2f} |")

    missing = [t for t in ("A", "B_eds640", "C_eds_aug640") if t not in seen]
    if missing:
        lines.append("")
        lines.append(f"_尚缺：{', '.join(missing)}_")

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
