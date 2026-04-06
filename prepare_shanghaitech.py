from __future__ import annotations

import argparse
from pathlib import Path

from clip_knn_baseline.common import ensure_frame_directories, save_clip_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ShanghaiTech frames and clip lengths.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data") / "shanghaitech",
        help="Root directory for the ShanghaiTech dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if frame folders already exist.",
    )
    args = parser.parse_args()

    ensure_frame_directories(args.dataset_root, overwrite=args.overwrite)
    save_clip_lengths(args.dataset_root)
    print(f"Prepared dataset at {args.dataset_root}")


if __name__ == "__main__":
    main()
