"""Sanity check for the project environment.

    python src/check_env.py

Prints the interpreter, the deep-learning stack, and runs one tiny convolution
on the GPU so that a broken CUDA install surfaces here instead of in training.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=" * 62)
    print("interpreter")
    print("=" * 62)
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  executable  : {sys.executable}")
    print(f"  platform    : {platform.platform()}")

    missing = []

    def version_of(module_name: str, label: str | None = None) -> str | None:
        try:
            mod = __import__(module_name)
        except Exception as exc:  # noqa: BLE001 - report anything
            print(f"  {label or module_name:<11} : MISSING ({exc.__class__.__name__})")
            missing.append(module_name)
            return None
        ver = getattr(mod, "__version__", "?")
        print(f"  {label or module_name:<11} : {ver}")
        return ver

    print("=" * 62)
    print("packages")
    print("=" * 62)
    version_of("torch")
    version_of("torchvision")
    version_of("numpy")
    version_of("cv2", "opencv")
    version_of("pandas")
    version_of("PIL", "pillow")
    version_of("yaml", "pyyaml")
    version_of("matplotlib")
    version_of("tqdm")
    version_of("scipy")
    yolo_version = version_of("ultralytics")

    print("=" * 62)
    print("cuda")
    print("=" * 62)
    cuda_ok = False
    try:
        import torch

        print(f"  torch cuda build   : {torch.version.cuda}")
        print(f"  cudnn              : {torch.backends.cudnn.version()}")
        print(f"  torch.cuda.available: {torch.cuda.is_available()}")
        print(f"  device count       : {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                print(f"  gpu[{i}]             : {props.name}")
                print(f"      capability     : sm_{props.major}{props.minor}")
                print(f"      vram           : {total / 1024**3:.1f} GiB "
                      f"({free / 1024**3:.1f} GiB free)")
            x = torch.randn(512, 512, device="cuda")
            w = torch.randn(512, 512, device="cuda")
            y = (x @ w).sum().item()
            torch.cuda.synchronize()
            print(f"  matmul on gpu      : OK (sum={y:.2f})")
            cuda_ok = True
        else:
            print("  !! torch cannot see a CUDA device")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! cuda check failed: {exc.__class__.__name__}: {exc}")

    print("=" * 62)
    if missing:
        print(f"missing packages: {', '.join(missing)}")
    if cuda_ok and yolo_version:
        print("RESULT: environment ready")
        return 0
    print("RESULT: environment NOT ready")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
