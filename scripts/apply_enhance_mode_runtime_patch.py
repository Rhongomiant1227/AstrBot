from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the versioned enhance-mode runtime patch to a plugin main.py file."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the runtime plugin main.py file to overwrite.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "runtime_patches"
        / "astrbot_plugin_astrbot_enhance_mode"
        / "main.py"
    )
    target = Path(args.target).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Patch source not found: {source}")
    if not target.exists():
        raise FileNotFoundError(f"Target file not found: {target}")
    if target.name != "main.py":
        raise ValueError(f"Target must be a main.py file: {target}")

    backup = target.with_name(
        f"{target.stem}.bak_repo_patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}"
    )
    shutil.copy2(target, backup)
    shutil.copy2(source, target)

    print(f"source={source}")
    print(f"backup={backup}")
    print(f"patched={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
