from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import faiss
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from clip_knn_baseline.common import load_test_ground_truth, macro_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ShanghaiTech frames with a CLIP kNN index.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data") / "shanghaitech",
        help="Root directory for the ShanghaiTech dataset.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("artifacts") / "clip_knn_baseline" / "shanghaitech",
        help="Directory containing extracted train and test features.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors used per region.",
    )
    parser.add_argument(
        "--max-train-features",
        type=int,
        default=200000,
        help="Randomly subsample the train feature bank to this many rows; set to 0 to keep all features.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for train-bank subsampling.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts") / "clip_knn_baseline" / "shanghaitech" / "results.json",
        help="Where to write summary metrics.",
    )
    parser.add_argument(
        "--save-scores-path",
        type=Path,
        default=Path("artifacts") / "clip_knn_baseline" / "shanghaitech" / "test_frame_scores.npy",
        help="Where to save per-frame anomaly scores.",
    )
    return parser.parse_args()


def load_feature_bank(features_path: Path) -> List[np.ndarray]:
    feature_list = np.load(features_path, allow_pickle=True)
    return [np.asarray(item, dtype=np.float32) for item in feature_list]


def flatten_feature_bank(feature_list: Sequence[np.ndarray]) -> np.ndarray:
    non_empty = [item for item in feature_list if item.size > 0]
    if not non_empty:
        raise RuntimeError("No features were found in the requested split.")
    return np.concatenate(non_empty, axis=0).astype(np.float32)


def maybe_subsample(features: np.ndarray, max_features: int, seed: int) -> np.ndarray:
    if max_features <= 0 or features.shape[0] <= max_features:
        return features
    rng = np.random.default_rng(seed)
    indices = rng.choice(features.shape[0], size=max_features, replace=False)
    return features[indices]


def build_index(train_features: np.ndarray) -> faiss.Index:
    dimension = train_features.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(train_features.astype(np.float32))
    return index


def frame_scores_from_knn(index: faiss.Index, feature_list: Sequence[np.ndarray], neighbors: int) -> np.ndarray:
    scores: List[float] = []
    k = min(neighbors, index.ntotal)
    if k < 1:
        raise RuntimeError("The kNN index is empty.")

    for frame_features in tqdm(feature_list, desc="Scoring test frames"):
        if frame_features.size == 0:
            scores.append(0.0)
            continue
        similarities, _ = index.search(frame_features.astype(np.float32), k)
        object_scores = 1.0 - similarities.mean(axis=1)
        scores.append(float(object_scores.max()))
    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    args = parse_args()
    train_features = load_feature_bank(args.features_dir / "train" / "features.npy")
    test_features = load_feature_bank(args.features_dir / "test" / "features.npy")

    train_matrix = flatten_feature_bank(train_features)
    train_matrix = maybe_subsample(train_matrix, args.max_train_features, args.seed)

    index = build_index(train_matrix)
    scores = frame_scores_from_knn(index=index, feature_list=test_features, neighbors=args.neighbors)

    args.save_scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.save_scores_path, scores)

    results = {
        "neighbors": args.neighbors,
        "train_features_indexed": int(index.ntotal),
        "test_frames_scored": int(scores.shape[0]),
    }

    try:
        labels, clip_lengths = load_test_ground_truth(args.dataset_root)
        if labels.shape[0] != scores.shape[0]:
            raise RuntimeError(
                f"Label/frame mismatch: loaded {labels.shape[0]} labels but scored {scores.shape[0]} frames."
            )
        results["micro_auc"] = float(roc_auc_score(labels, scores))
        results["macro_auc"] = float(macro_auc(scores, labels, clip_lengths, roc_auc_score))
    except FileNotFoundError:
        results["note"] = "Ground-truth test_frame_mask files were not found, so AUC was skipped."

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
