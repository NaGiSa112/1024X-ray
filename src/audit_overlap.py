"""How much do weak-class boxes overlap boxes of *other* classes?

Loose or duplicate annotations show up as a box that covers most of another
class's box.  For each weak class, report the share of boxes whose best IoU with
any other-class box in the same image is above 0.5 / 0.3, split by the box being
"extreme large" (>3x median area) or not.

High overlap among the extreme-large boxes is evidence that those annotations are
not a separate object but a loose/misplaced box, which caps AP no matter how big
the model gets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "annotations.csv", dtype={"image": str})
df["area"] = ((df.x2 - df.x1) / df.width) * ((df.y2 - df.y1) / df.height)

WEAK = ["pressuretank", "tool", "drinkbottle"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


by_image = {img: sub for img, sub in df.groupby("image")}

for cls in WEAK:
    g = df[df["class"] == cls]
    med = g.area.median()
    print("=" * 66)
    print(f"{cls}: {len(g)} boxes, median area {med:.4f}")

    groups = {
        "extreme large (>3x median)": g[g.area > med * 3],
        "normal": g[(g.area >= med / 3) & (g.area <= med * 3)],
        "extreme small (<1/3 median)": g[g.area < med / 3],
    }
    print(f"{'group':<30}{'boxes':>7}{'IoU>0.5':>10}{'IoU>0.3':>10}{'median IoU':>12}")
    for label, sub in groups.items():
        best = []
        for _, r in sub.iterrows():
            others = by_image[r["image"]]
            others = others[others["class"] != cls]
            if len(others) == 0:
                best.append(0.0)
                continue
            box = np.array([[r.x1, r.y1, r.x2, r.y2]], dtype=float)
            m = iou_matrix(box, others[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float))
            best.append(float(m.max()))
        best_arr = np.asarray(best)
        if len(best_arr) == 0:
            continue
        print(f"{label:<30}{len(sub):>7}{(best_arr > 0.5).mean():>9.1%}"
              f"{(best_arr > 0.3).mean():>10.1%}{np.median(best_arr):>12.3f}")
