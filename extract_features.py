from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from clip_knn_baseline.common import FrameRecord, collect_frame_records


DEFAULT_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "skateboard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP features from YOLO detections.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data") / "shanghaitech",
        help="Root directory for the ShanghaiTech dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "clip_knn_baseline" / "shanghaitech",
        help="Directory used to save extracted detections and features.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=("train", "test", "both"),
        help="Which split to process.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics detector checkpoint.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP image encoder name.",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weights tag.",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=16 * 16,
        help="Minimum area for accepted detections.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.05,
        help="Extra padding added around each box as a ratio of box width and height.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated COCO class names to keep.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=32,
        help="Batch size for CLIP encoding.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Detector inference size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--video-limit",
        type=int,
        default=0,
        help="Optional limit on the number of videos per split for quick smoke tests.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=0,
        help="Optional limit on the number of frames per split for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing feature dump.",
    )
    parser.add_argument(
        "--no-full-frame-fallback",
        action="store_true",
        help="Disable full-frame CLIP fallback when no detections are found.",
    )
    return parser.parse_args()


def resolve_class_ids(detector: YOLO, class_names: Sequence[str]) -> List[int]:
    detector_names = detector.model.names
    normalized = {str(name).strip().lower(): idx for idx, name in detector_names.items()}
    class_ids = []
    missing = []
    for class_name in class_names:
        key = class_name.strip().lower()
        if key in normalized:
            class_ids.append(normalized[key])
        else:
            missing.append(class_name)
    if missing:
        raise ValueError(f"Unknown detector classes: {', '.join(missing)}")
    return class_ids


def clip_encode(
    clip_model: torch.nn.Module,
    preprocess,
    device: str,
    images: Sequence[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    encoded_batches: List[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        tensors = []
        for image_rgb in batch_images:
            pil_image = Image.fromarray(image_rgb)
            tensors.append(preprocess(pil_image))
        batch_tensor = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        encoded_batches.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(encoded_batches, axis=0)


def crop_box(image_bgr: np.ndarray, box: np.ndarray, padding_ratio: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(np.float32)
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(width, int(np.ceil(x2 + pad_x)))
    y2 = min(height, int(np.ceil(y2 + pad_y)))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def limit_records(records: Sequence[FrameRecord], video_limit: int, frame_limit: int) -> List[FrameRecord]:
    if video_limit > 0:
        keep_videos = []
        for record in records:
            if record.video_name not in keep_videos:
                keep_videos.append(record.video_name)
            if len(keep_videos) == video_limit:
                break
        records = [record for record in records if record.video_name in set(keep_videos)]
    if frame_limit > 0:
        records = list(records[:frame_limit])
    return list(records)


def save_metadata(output_path: Path, metadata_rows: Sequence[Tuple[int, str, int, str, int]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "video_name", "frame_index", "frame_path", "num_regions"])
        writer.writerows(metadata_rows)


def extract_split(args: argparse.Namespace, detector: YOLO, clip_model: torch.nn.Module, preprocess, split: str) -> None:
    split_output_dir = args.output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    features_path = split_output_dir / "features.npy"
    boxes_path = split_output_dir / "boxes.npy"
    classes_path = split_output_dir / "classes.npy"
    metadata_path = split_output_dir / "metadata.csv"
    if features_path.exists() and boxes_path.exists() and classes_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"Skipping {split}: cached features already exist at {split_output_dir}")
        return

    class_names = [item.strip() for item in args.classes.split(",") if item.strip()]
    class_ids = resolve_class_ids(detector, class_names)
    frame_records, _ = collect_frame_records(args.dataset_root, split)
    frame_records = limit_records(frame_records, args.video_limit, args.frame_limit)

    per_frame_features: List[np.ndarray] = []
    per_frame_boxes: List[np.ndarray] = []
    per_frame_classes: List[np.ndarray] = []
    metadata_rows: List[Tuple[int, str, int, str, int]] = []

    for frame_id, record in enumerate(tqdm(frame_records, desc=f"Extracting {split} features")):
        image_bgr = cv2.imread(str(record.frame_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not read frame: {record.frame_path}")

        result = detector.predict(
            source=image_bgr,
            conf=args.det_conf,
            classes=class_ids,
            imgsz=args.imgsz,
            verbose=False,
            device=args.device,
        )[0]

        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else np.empty((0, 4), dtype=np.float32)
        labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64) if result.boxes is not None else np.empty((0,), dtype=np.int64)

        if boxes.shape[0] > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep_mask = (widths * heights) >= args.min_box_area
            boxes = boxes[keep_mask]
            labels = labels[keep_mask]

        crops: List[np.ndarray] = []
        if boxes.shape[0] > 0:
            for box in boxes:
                crops.append(crop_box(image_bgr, box, args.padding_ratio))
        elif not args.no_full_frame_fallback:
            height, width = image_bgr.shape[:2]
            boxes = np.asarray([[0, 0, width, height]], dtype=np.float32)
            labels = np.asarray([-1], dtype=np.int64)
            crops.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if crops:
            features = clip_encode(
                clip_model=clip_model,
                preprocess=preprocess,
                device=args.device,
                images=crops,
                batch_size=args.clip_batch_size,
            )
        else:
            clip_dim = clip_model.visual.output_dim
            features = np.empty((0, clip_dim), dtype=np.float32)

        per_frame_features.append(features)
        per_frame_boxes.append(boxes.astype(np.float32))
        per_frame_classes.append(labels.astype(np.int64))
        metadata_rows.append(
            (
                frame_id,
                record.video_name,
                record.frame_index,
                str(record.frame_path),
                int(features.shape[0]),
            )
        )

    np.save(features_path, np.asarray(per_frame_features, dtype=object))
    np.save(boxes_path, np.asarray(per_frame_boxes, dtype=object))
    np.save(classes_path, np.asarray(per_frame_classes, dtype=object))
    save_metadata(metadata_path, metadata_rows)
    print(f"Saved {split} features to {split_output_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    clip_model = clip_model.to(args.device)
    clip_model.eval()

    detector = YOLO(args.detector)

    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        extract_split(args, detector, clip_model, preprocess, split)


if __name__ == "__main__":
    main()
