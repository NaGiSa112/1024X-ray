"""Canonical class definition for the X-ray detection task.

The eight class names are what appear in `data/train/XML/*.xml`.  The order
below is **alphabetical** -- it is a neutral, reproducible choice for training,
NOT a claim about the competition's submission order.

⚠️ The submission template `results_test2.json` is positional: entry `i` of each
image is a list of boxes for "class i", but the template carries no names, so
the real order can only be read off the competition page ("赛题数据 / 提交说明").
Everything that writes a prediction must therefore go through
`SUBMISSION_ORDER`, so that reordering it in one place is enough.
"""

from __future__ import annotations

# Training order (alphabetical).
CLASSES: list[str] = [
    "drinkbottle",
    "glassbottle",
    "knife",
    "laptop",
    "metalbottle",
    "pressuretank",
    "tool",
    "umbrella",
]

# Order used when writing `results_test2.json`.  Confirmed against the official
# competition data interface (challenge.xfyun.cn/2020/ai-contest/api/page-data/
# Xray-25-train) on 2026-11-20 -- see 仓库链接与首次训练方案.md section 2.
# Note this is NOT the training order above: e.g. training id 2 (knife) is
# submission slot 0, and training id 1 (glassbottle) is slot 2.
SUBMISSION_ORDER: list[str] = [
    "knife",         # slot 0
    "tool",          # slot 1
    "glassbottle",   # slot 2
    "drinkbottle",   # slot 3
    "metalbottle",   # slot 4
    "laptop",        # slot 5
    "umbrella",      # slot 6
    "pressuretank",  # slot 7
]

NAME_TO_ID = {name: i for i, name in enumerate(CLASSES)}
ID_TO_NAME = {i: name for i, name in enumerate(CLASSES)}

# training id -> submission slot, for exporting results_test2.json
TRAIN_ID_TO_SLOT: dict[int, int] = {
    NAME_TO_ID[name]: slot for slot, name in enumerate(SUBMISSION_ORDER)
}
# name -> submission slot (safer: go through model.names, never raw ids)
NAME_TO_SLOT: dict[str, int] = {name: slot for slot, name in enumerate(SUBMISSION_ORDER)}

# Chinese labels, for plots and reports.  Purely cosmetic.
ZH_NAME: dict[str, str] = {
    "drinkbottle": "饮料瓶",
    "glassbottle": "玻璃瓶",
    "knife": "刀具",
    "laptop": "笔记本电脑",
    "metalbottle": "金属瓶",
    "pressuretank": "压力罐",
    "tool": "工具",
    "umbrella": "雨伞",
}

NUM_CLASSES = len(CLASSES)
