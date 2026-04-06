# Frame-Based YOLO + CLIP + kNN Baseline

This baseline keeps the pipeline intentionally simple:

1. Extract frames if the split only contains raw videos.
2. Detect foreground objects with YOLO.
3. Encode each object crop with CLIP.
4. Build a kNN index from train features only.
5. Score each test frame by the most anomalous detected region in that frame.

There is no temporal smoothing, tracking, or optical flow in this version.

## Expected dataset layout

The scripts expect the ShanghaiTech dataset under `data/shanghaitech` with this structure:

```text
data/shanghaitech/
  training/
    videos/      # optional if frames already exist
    frames/
  testing/
    videos/      # optional if frames already exist
    frames/
    test_frame_mask/
```

If the Kaggle train and test splits extract into nested folders, move the contents so they match this layout before running the scripts.

## Colab setup

In Google Colab, a typical setup looks like this:

```bash
git clone <your-repo-url>
cd Accurate-Interpretable-VAD
pip install -r clip_knn_baseline/requirements-colab.txt
```

If you want a single self-contained notebook instead, use [shanghaitech_clip_knn_colab.ipynb](./shanghaitech_clip_knn_colab.ipynb).

If you want to download from Kaggle inside Colab:

```python
from google.colab import files
files.upload()  # upload kaggle.json
```

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d nikanvasei/shanghaitech-campus-dataset -p /content/datasets
kaggle datasets download -d nikanvasei/shanghaitech-campus-dataset-test -p /content/datasets
mkdir -p data/shanghaitech
unzip -q /content/datasets/shanghaitech-campus-dataset.zip -d data/shanghaitech
unzip -q /content/datasets/shanghaitech-campus-dataset-test.zip -d data/shanghaitech
```

After unzipping, make sure the final folders line up with the expected layout above.

## 1. Prepare frames

This extracts frames from raw videos when needed and writes `train_clip_lengths.npy` and `test_clip_lengths.npy`.

```bash
python -m clip_knn_baseline.prepare_shanghaitech --dataset-root data/shanghaitech
```

## 2. Extract YOLO + CLIP features

The default configuration uses `yolov8n.pt` and `ViT-B-32`.

```bash
python -m clip_knn_baseline.extract_features \
  --dataset-root data/shanghaitech \
  --output-dir artifacts/clip_knn_baseline/shanghaitech \
  --split both \
  --detector yolov8n.pt \
  --device cuda
```

Useful knobs:

- `--video-limit 2` for quick smoke tests
- `--frame-limit 500` for quick smoke tests
- `--detector yolov8s.pt` for a stronger detector
- `--no-full-frame-fallback` if you want pure object-only scoring

The extractor saves:

- `artifacts/clip_knn_baseline/shanghaitech/train/features.npy`
- `artifacts/clip_knn_baseline/shanghaitech/test/features.npy`
- `artifacts/clip_knn_baseline/shanghaitech/<split>/boxes.npy`
- `artifacts/clip_knn_baseline/shanghaitech/<split>/classes.npy`
- `artifacts/clip_knn_baseline/shanghaitech/<split>/metadata.csv`

## 3. Score test frames with kNN

```bash
python -m clip_knn_baseline.score_knn \
  --dataset-root data/shanghaitech \
  --features-dir artifacts/clip_knn_baseline/shanghaitech \
  --neighbors 5
```

Outputs:

- `artifacts/clip_knn_baseline/shanghaitech/test_frame_scores.npy`
- `artifacts/clip_knn_baseline/shanghaitech/results.json`

If `testing/test_frame_mask` exists, the scorer also reports micro and macro frame-level AUC.

## Notes

- This is a solid baseline, but it will be weaker on anomalies that are mostly defined by motion rather than appearance.
- Colab GPU is mainly used by YOLO and CLIP. The FAISS step uses a CPU index for portability.
- If memory becomes tight, reduce the train bank with `--max-train-features`.
