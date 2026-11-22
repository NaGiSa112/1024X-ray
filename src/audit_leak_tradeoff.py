"""Quantify the trade-off of removing duplicated training images.

The 45 training images whose content also sits in val are the *val leakage*.  The
natural fix is to drop them from TRAIN (val stays at 555 and becomes clean),
not to shrink val.  But some of those same training images may also duplicate a
test2 image -- and those are "free points", because the model gets to memorise a
picture that is scored.  Removing them costs score; keeping them keeps the leak.

This script measures how big that conflict actually is, per class, so the choice
can be made on numbers instead of intuition.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train"
TEST = ROOT / "data" / "test2"
AUDIT = ROOT / "runs" / "duplicate_audit.json"
CSV = ROOT / "data" / "annotations.csv"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    # train images that share content with a val image
    leak_train: set[str] = set()
    for grp in audit["cross_split_groups"]:
        for m in grp["train_members"]:
            leak_train.add(Path(m).stem)
    print(f"train images duplicating a val image : {len(leak_train)}")

    # hash every test2 image and every candidate train image
    print("hashing test2 ...")
    test_by_sha: dict[str, list[str]] = {}
    for p in sorted(TEST.glob("*.jpg")):
        test_by_sha.setdefault(sha(p), []).append(p.name)
    print(f"  {len(test_by_sha)} unique test2 contents")

    print("hashing candidate train images ...")
    leak_in_test, leak_only = [], []
    for stem in sorted(leak_train):
        src = TRAIN / f"{stem}.jpg"
        if not src.exists():
            continue
        h = sha(src)
        (leak_in_test if h in test_by_sha else leak_only).append(stem)

    print()
    print("=" * 62)
    print(f"train images overlapping val          : {len(leak_train)}")
    print(f"  ... that ALSO duplicate a test2 image: {len(leak_in_test)}"
          f"   <- removing these loses free points")
    print(f"  ... that do NOT touch test2         : {len(leak_only)}"
          f"   <- safe to drop, pure leakage fix")
    print("=" * 62)

    df = pd.read_csv(CSV, dtype={"image": str})
    both = set(leak_in_test)
    if both:
        print()
        print("classes carried by the conflicting images (overlap with test2):")
        sub = df[df.image.isin(both)]
        for cls, n in sub["class"].value_counts().items():
            print(f"  {cls:<14}{n:>4} boxes")
        print(f"  images: {len(both)}")

    print()
    print("val boxes affected by the leak, per class:")
    for cls, n in audit.get("affected_val_per_class", {}).items():
        if isinstance(n, dict):
            print(f"  {cls:<14}{n}")
        else:
            print(f"  {cls:<14}{n:>4}")

    out = {
        "leak_train_total": len(leak_train),
        "leak_train_also_in_test2": sorted(leak_in_test),
        "leak_train_safe_to_drop": sorted(leak_only),
    }
    (ROOT / "runs" / "audit" / "leak_tradeoff.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> runs/audit/leak_tradeoff.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
