"""Build contact sheets of the weak-class boxes so the labels can be eyeballed.

    python src/audit_weak_classes.py --per-class 30 --out .tmp/audit

One sheet per class: each cell is the annotated box, cropped and resized to a
fixed size.  If a class is labelled inconsistently (two different objects sharing
one name, or a subclass silently included), it shows up immediately as visual
clusters that do not belong together -- which no amount of model capacity or
augmentation can fix.

Also prints per-class box geometry so suspicious structure (bimodal sizes,
near-duplicate boxes) is visible in numbers too.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "annotations.csv"
TRAIN = ROOT / "data" / "train"

WEAK = ["pressuretank", "tool", "drinkbottle"]


def sheet_for(df: pd.DataFrame, cls: str, n: int, out: Path, cell: int = 128, cols: int = 6) -> Path:
    rows = df[df["class"] == cls].sample(n=min(n, (df["class"] == cls).sum()), random_state=0)
    rows = rows.sort_values("image")
    n_cells = len(rows)
    grid_rows = (n_cells + cols - 1) // cols
    pad = 4
    canvas = Image.new("RGB", (cols * (cell + pad) + pad, grid_rows * (cell + pad) + pad), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    for k, (_, r) in enumerate(rows.iterrows()):
        img = Image.open(TRAIN / f"{r['image']}.jpg").convert("RGB")
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        if crop.width < 2 or crop.height < 2:
            continue
        crop.thumbnail((cell, cell), Image.LANCZOS)
        cx = pad + (k % cols) * (cell + pad)
        cy = pad + (k // cols) * (cell + pad)
        canvas.paste(crop, (cx + (cell - crop.width) // 2, cy + (cell - crop.height) // 2))
        draw.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], outline=(70, 70, 70))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / ".tmp" / "audit"))
    args = ap.parse_args()

    # dtype keeps the zero-padded stems ("00013") that pandas would otherwise
    # parse as the integer 13
    df = pd.read_csv(CSV, dtype={"image": str})
    df["bw"] = (df.x2 - df.x1) / df.width
    df["bh"] = (df.y2 - df.y1) / df.height
    df["area"] = df.bw * df.bh
    df["ar"] = df.bw / df.bh

    out_dir = Path(args.out)
    for cls in WEAK:
        g = df[df["class"] == cls]
        print("=" * 62)
        print(f"{cls}: {len(g)} boxes in {g.image.nunique()} images")
        print(f"  面积(归一化)  中位 {g.area.median():.4f}  p5 {g.area.quantile(.05):.4f}  "
              f"p95 {g.area.quantile(.95):.4f}")
        print(f"  长宽比        中位 {g.ar.median():.2f}  p5 {g.ar.quantile(.05):.2f}  "
              f"p95 {g.ar.quantile(.95):.2f}")
        # bimodality hint: share of boxes in the small vs large half
        small = (g.area < g.area.median() / 3).mean()
        large = (g.area > g.area.median() * 3).mean()
        print(f"  极端小框占比 {small:.1%}   极端大框占比 {large:.1%}")
        # duplicate boxes: same image, same class, IoU high
        dup = 0
        for _, sub in g.groupby("image"):
            vals = sub[["x1", "y1", "x2", "y2"]].to_numpy()
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    ix1, iy1 = max(vals[i][0], vals[j][0]), max(vals[i][1], vals[j][1])
                    ix2, iy2 = min(vals[i][2], vals[j][2]), min(vals[i][3], vals[j][3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter <= 0:
                        continue
                    a1 = (vals[i][2] - vals[i][0]) * (vals[i][3] - vals[i][1])
                    a2 = (vals[j][2] - vals[j][0]) * (vals[j][3] - vals[j][1])
                    if inter / (a1 + a2 - inter) > 0.85:
                        dup += 1
        print(f"  近乎重复的同类框对: {dup}")
        # how often does this class co-occur with another bottle class?
        co = Counter()
        for _, sub in g.groupby("image"):
            others = df[(df.image == sub.image.iloc[0]) & (df["class"] != cls)]["class"]
            for o in set(others):
                co[o] += 1
        print(f"  最常见共现类别: {co.most_common(4)}")
        p = sheet_for(df, cls, args.per_class, out_dir / f"{cls}.png")
        print(f"  样图 -> {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
