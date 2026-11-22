"""Probe: can the jgen DINO (EDS-1) checkpoint actually run inference here?

    E:\\conda_envs\\dino_teacher\\python.exe src\\dino_teacher_probe.py
    E:\\conda_envs\\dino_teacher\\python.exe src\\dino_teacher_probe.py --val-n 20

Why DINO and not Co-DETR: the Co-DETR weights are pinned to torch 1.11 + cu113,
which has no kernels for the RTX 4090's sm_89, so that checkpoint cannot run on
this machine at all.  DINO is pinned to torch 2.1.1 + cu121, which does.

The point of this probe is not accuracy yet -- it is to prove the whole stack
(mmdet config -> checkpoint load -> deformable attention CUDA kernel -> boxes)
works before spending time on evaluation.

Class mapping, since EDS-1 has 10 classes and this task has 8:

    EDS-1                    -> ours
    knife                      knife          (exact)
    laptop                     laptop         (exact)
    umbrella                   umbrella       (exact)
    glassbottle                glassbottle    (exact)
    pressure                   pressuretank   (high confidence)
    plasticbottle              drinkbottle    (partial: soft-drink vs plastic bottle)
    scissor, device            tool           (partial)
    lighter, powerbank         (dropped, no counterpart)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(r"E:\X光")
CFG = ROOT / "deps" / "jgen_repo" / "dino" / "cfg" / "dino-5scale-swin-b-8xb2-36e_EDS_1_2.py"
CKPT = ROOT / "weights" / "dino_eds1.pth"
VAL_DIR = ROOT / "data" / "yolo_clean" / "images" / "val"

EDS_CLASSES = ["plasticbottle", "knife", "scissor", "laptop", "umbrella",
               "lighter", "device", "powerbank", "pressure", "glassbottle"]
MAP = {"knife": "knife", "laptop": "laptop", "umbrella": "umbrella",
       "glassbottle": "glassbottle", "pressure": "pressuretank",
       "plasticbottle": "drinkbottle", "scissor": "tool", "device": "tool"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-n", type=int, default=3, help="how many val images to run")
    ap.add_argument("--score-thr", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")

    # 1) deformable attention kernel on sm_89 (informational: the real check is
    #    step 2, which exercises the same kernel inside the loaded model)
    from mmcv.ops import MultiScaleDeformableAttention
    try:
        mod = MultiScaleDeformableAttention(embed_dims=64, num_heads=4, num_levels=1,
                                            num_points=4, batch_first=True).cuda()
        q = torch.randn(1, 32 * 32, 64, device="cuda")
        ref = torch.rand(1, 32 * 32, 1, 2, device="cuda")
        val = torch.randn(1, 32 * 32, 64, device="cuda")
        spatial = torch.as_tensor([[32, 32]], dtype=torch.long, device="cuda")
        lsi = torch.as_tensor([0], dtype=torch.long, device="cuda")
        out = mod(q, ref, val, spatial, lsi)
        print(f"[1] mmcv deformable attention on GPU: OK {tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[1] synthetic deform-attn check skipped ({type(exc).__name__}: "
              f"{str(exc)[:80]}) -- real kernel check happens in step 3")

    # 2) build the model from their config and load the checkpoint
    from mmdet.apis import init_detector, inference_detector
    from mmdet.registry import DATASETS
    if not CFG.is_file():
        print(f"missing config: {CFG}")
        return 1
    if not CKPT.is_file():
        print(f"missing checkpoint: {CKPT}")
        return 1
    print(f"[2] building model from {CFG.name} ...")
    model = init_detector(str(CFG), str(CKPT), device="cuda:0", palette="none")
    head = getattr(model, "bbox_head", None)
    print(f"    model class : {type(model).__name__}")
    print(f"    num_classes : {getattr(head, 'num_classes', '?')}")
    print(f"[2] checkpoint loaded OK")

    # 3) real inference on the task's own validation images
    import numpy as np
    imgs = sorted(VAL_DIR.glob("*.jpg"))[: args.val_n]
    print(f"[3] running inference on {len(imgs)} image(s)")
    for p in imgs:
        res = inference_detector(model, str(p))
        pred = res.pred_instances
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()
        keep = scores >= args.score_thr
        print(f"\n    {p.name}: {int(keep.sum())} boxes @>{args.score_thr}")
        if keep.sum():
            per = {}
            for lab, sc in zip(labels[keep], scores[keep]):
                name = EDS_CLASSES[int(lab)] if int(lab) < len(EDS_CLASSES) else f"cls{lab}"
                ours = MAP.get(name, f"--{name}--")
                per.setdefault(ours, []).append(float(sc))
            for k, v in sorted(per.items(), key=lambda kv: -max(kv[1])):
                print(f"      {k:<14} n={len(v):3d}  max={max(v):.3f}  mean={np.mean(v):.3f}")
        else:
            print("      (no detections above threshold)")
    print("\nPROBE OK: DINO EDS-1 runs on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
