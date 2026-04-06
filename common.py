from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


SPLIT_TO_DIR = {
    "train": "training",
    "test": "testing",
}


@dataclass(frozen=True)
class FrameRecord:
    video_name: str
    frame_index: int
    frame_path: Path


def _frame_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (10**9, stem)


def _iter_frame_files(video_dir: Path) -> Iterable[Path]:
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in video_dir.glob(pattern):
            yield path


def collect_frame_records(dataset_root: Path | str, split: str) -> Tuple[List[FrameRecord], np.ndarray]:
    dataset_root = Path(dataset_root)
    split_dir = SPLIT_TO_DIR[split]
    frames_root = dataset_root / split_dir / "frames"
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_root}")

    frame_records: List[FrameRecord] = []
    clip_lengths: List[int] = []
    total = 0

    for video_dir in sorted([path for path in frames_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        frames = sorted(_iter_frame_files(video_dir), key=_frame_sort_key)
        for frame_index, frame_path in enumerate(frames):
            frame_records.append(FrameRecord(video_name=video_dir.name, frame_index=frame_index, frame_path=frame_path))
        total += len(frames)
        clip_lengths.append(total)

    return frame_records, np.asarray(clip_lengths, dtype=np.int64)


def save_clip_lengths(dataset_root: Path | str) -> None:
    dataset_root = Path(dataset_root)
    for split in ("train", "test"):
        try:
            _, clip_lengths = collect_frame_records(dataset_root, split)
        except FileNotFoundError:
            continue
        output_name = "train_clip_lengths.npy" if split == "train" else "test_clip_lengths.npy"
        np.save(dataset_root / output_name, clip_lengths)


def extract_frames_from_video(video_path: Path, output_dir: Path, overwrite: bool = False) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("*.jpg"), key=_frame_sort_key)
    if existing_frames and not overwrite:
        return len(existing_frames)

    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = 0
    success, frame = capture.read()
    while success:
        cv2.imwrite(str(output_dir / f"{frame_count}.jpg"), frame)
        frame_count += 1
        success, frame = capture.read()

    capture.release()
    return frame_count


def ensure_frame_directories(dataset_root: Path | str, overwrite: bool = False) -> None:
    dataset_root = Path(dataset_root)

    for split, split_dir in SPLIT_TO_DIR.items():
        videos_root = dataset_root / split_dir / "videos"
        frames_root = dataset_root / split_dir / "frames"

        if not videos_root.exists():
            continue

        frames_root.mkdir(parents=True, exist_ok=True)
        for video_path in sorted([path for path in videos_root.iterdir() if path.is_file()], key=lambda path: path.name):
            extract_frames_from_video(video_path, frames_root / video_path.stem, overwrite=overwrite)

    save_clip_lengths(dataset_root)


def load_test_ground_truth(dataset_root: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    dataset_root = Path(dataset_root)
    masks_root = dataset_root / "testing" / "test_frame_mask"
    if not masks_root.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {masks_root}")

    mask_files = sorted(masks_root.glob("*.npy"), key=lambda path: path.name)
    all_labels: List[np.ndarray] = []
    clip_lengths: List[int] = []
    total = 0
    for mask_path in mask_files:
        labels = np.load(mask_path)
        labels = labels.astype(np.int64).reshape(-1)
        all_labels.append(labels)
        total += labels.shape[0]
        clip_lengths.append(total)

    if not all_labels:
        raise RuntimeError(f"No frame-mask files found in: {masks_root}")

    return np.concatenate(all_labels, axis=0), np.asarray(clip_lengths, dtype=np.int64)


def macro_auc(scores: Sequence[float], labels: Sequence[int], clip_lengths: Sequence[int], roc_auc_score) -> float:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    prev = 0
    aucs: List[float] = []
    for end in clip_lengths:
        video_scores = scores[prev:end]
        video_labels = labels[prev:end]
        padded_labels = np.concatenate(([0], video_labels, [1]))
        padded_scores = np.concatenate(([0.0], video_scores, [sys.float_info.max]))
        aucs.append(float(roc_auc_score(padded_labels, padded_scores)))
        prev = end

    if not aucs:
        raise RuntimeError("Macro AUC could not be computed because no per-video scores were available.")
    return float(np.mean(aucs))
