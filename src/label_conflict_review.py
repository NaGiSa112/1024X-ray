"""Classify duplicate-image annotation conflicts by severity, and render only
the ones that actually need human eyes.

    python src/label_conflict_review.py            # classify + summary
    python src/label_conflict_review.py --render   # also export side-by-side PNGs

Why this exists: the earlier audit marks a duplicate group "inconsistent" if the
pixel coordinates differ at all, which lumps a 1-pixel jitter together with a
genuinely different box set.  Sorting by severity turns "215 groups need review"
into "N groups need review".

Severity:
    identical          same classes, every box within 2 px
    jitter             same class counts, max deviation <= 8 px  -> either copy is fine
    coordinate_mismatch same class counts, deviation > 8 px      -> one copy is looser
    class_mismatch     class counts differ                       -> real conflict

Outputs:
    runs/audit/conflicts/summary.json
    runs/audit/conflicts/summary.csv
    runs/audit/conflicts/<severity>/<index>.png   (with --render)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "train"          # images + XML live here
YOLO = ROOT / "data" / "yolo"
OUT = ROOT / "runs" / "audit" / "conflicts"

COLOURS = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 200, 60), (220, 120, 255)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_boxes(stem: str) -> list[tuple[str, float, float, float, float]]:
    xml = RAW / "XML" / f"{stem}.xml"
    if not xml.exists():
        return []
    root = ET.parse(xml).getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        out.append((name, float(box.findtext("xmin")), float(box.findtext("ymin")),
                    float(box.findtext("xmax")), float(box.findtext("ymax"))))
    return out


def compare(a: list, b: list) -> tuple[str, float, str]:
    """Return (severity, max_deviation, detail)."""
    ca, cb = Counter(x[0] for x in a), Counter(x[0] for x in b)
    if ca != cb:
        return "class_mismatch", float("inf"), f"{dict(ca)} vs {dict(cb)}"

    # greedy one-to-one match per class by nearest centre
    max_dev = 0.0
    for cls in ca:
        pa = [x for x in a if x[0] == cls]
        pb = [x for x in b if x[0] == cls]
        used = [False] * len(pb)
        for box_a in pa:
            best, best_j = float("inf"), -1
            for j, box_b in enumerate(pb):
                if used[j]:
                    continue
                d = max(abs(box_a[k] - box_b[k]) for k in (1, 2, 3, 4))
                if d < best:
                    best, best_j = d, j
            if best_j >= 0:
                used[best_j] = True
                max_dev = max(max_dev, best)
    if max_dev <= 2.0:
        return "identical", max_dev, f"max deviation {max_dev:.0f}px"
    if max_dev <= 8.0:
        return "jitter", max_dev, f"max deviation {max_dev:.0f}px"
    return "coordinate_mismatch", max_dev, f"max deviation {max_dev:.0f}px"


def render(group: list[dict], path: Path, max_side: int = 900) -> None:
    img_path = Path(group[0]["image_path"])
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        scale = max_side / max(im.size)
        if scale < 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        panels = []
        for gi, m in enumerate(group):
            canvas = im.copy()
            draw = ImageDraw.Draw(canvas)
            for bi, (cls, x1, y1, x2, y2) in enumerate(m["boxes"]):
                col = COLOURS[list(dict.fromkeys(b[0] for b in m["boxes"])).index(cls) % len(COLOURS)]
                box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
                draw.rectangle(box, outline=col, width=3)
                draw.text((box[0] + 3, max(0, box[1] - 12)), f"{cls}#{bi}", fill=col)
            draw.text((6, 6), f"{m['stem']}  ({len(m['boxes'])} boxes)", fill=(255, 255, 255))
            panels.append(canvas)
        w = sum(p.width for p in panels) + 12 * (len(panels) - 1)
        h = max(p.height for p in panels)
        sheet = Image.new("RGB", (w, h), (20, 20, 20))
        x = 0
        for p in panels:
            sheet.paste(p, (x, 0))
            x += p.width + 12
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    members = []
    for split in ("train", "val"):
        for p in sorted((YOLO / "images" / split).glob("*.jpg")):
            members.append((split, p))
    print(f"hashing {len(members)} images ...")
    by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for split, p in members:
        by_hash[sha(p)].append((split, p))

    groups = [(h, v) for h, v in by_hash.items() if len(v) > 1]
    print(f"duplicate content groups: {len(groups)}  "
          f"(redundant images: {sum(len(v) - 1 for _, v in groups)})")

    rows = []
    counter = Counter()
    for gi, (h, mem) in enumerate(groups):
        parsed = [{"split": s, "stem": p.stem, "image_path": str(p), "boxes": read_boxes(p.stem)}
                  for s, p in mem]
        base = parsed[0]["boxes"]
        worst_sev, worst_dev, worst_detail = "identical", 0.0, ""
        for other in parsed[1:]:
            sev, dev, detail = compare(base, other["boxes"])
            if sev == "class_mismatch" or dev > worst_dev:
                worst_sev, worst_dev, worst_detail = sev, dev, detail
        counter[worst_sev] += 1
        rows.append({
            "group": gi, "sha256": h[:16], "members": len(mem),
            "splits": ",".join(m["split"] for m in parsed),
            "stems": ",".join(m["stem"] for m in parsed),
            "box_counts": ",".join(str(len(m["boxes"])) for m in parsed),
            "severity": worst_sev, "max_dev_px": (None if worst_dev == float("inf") else round(worst_dev, 1)),
            "detail": worst_detail,
        })
        if args.render and worst_sev in ("class_mismatch", "coordinate_mismatch"):
            render(parsed, OUT / worst_sev / f"{gi:04d}_{'_'.join(m['stem'] for m in parsed)}.png")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"{'severity':<22}{'groups':>8}   处理建议")
    advice = {
        "identical": "无需处理，任选一份",
        "jitter": "无需人工，误差几像素",
        "coordinate_mismatch": "看并排图，选更贴边的一份",
        "class_mismatch": "必须人工判定",
    }
    for sev, n in counter.most_common():
        print(f"{sev:<22}{n:>8}   {advice[sev]}")
    need = counter.get("coordinate_mismatch", 0) + counter.get("class_mismatch", 0)
    print()
    print(f"=> 真正需要人工看的：{need} 组（其余 {len(groups) - need} 组可自动放行）")
    print(f"-> {OUT / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
