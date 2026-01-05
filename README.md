# apple-diameter-estimator 🍎📏

Estimate on-tree apple **diameter (cm)** from **RGB-D** imagery using a Detectron2-based pipeline with a custom regression head.

This repository contains the code developed for my Bachelor thesis project (Computer Engineering) focused on **apple diameter estimation** from **aligned RGB + depth** data acquired in orchard conditions.

---

## What this project does

Given an RGB image and its corresponding depth map (RGB-D), the system:

1. Detects (and optionally segments) apples in the RGB image (Detectron2).
2. Extracts ROI features for each detected apple.
3. Regresses a **single scalar**: the **apple diameter in centimeters**.

The overall goal is to provide an accurate and lightweight estimator that can run in real-world robotics settings (including embedded devices like NVIDIA Jetson).

---

## Approach (high level)

- **Backbone + detector**: Detectron2 (e.g., Faster R-CNN / Mask R-CNN style pipeline).
- **Input**: 4-channel tensor `[R, G, B, D]` (depth aligned to RGB).
- **Head**: custom ROI-level regressor that outputs a diameter value.
- **Targets**: ground-truth diameter (cm) attached to each apple annotation.

> Note: The exact architecture/config is defined in `configs/` and the custom modules in `custom/`.

---

## Repository structure

- `train.py` — training entrypoint
- `requirements.txt` — Python dependencies
- `configs/` — Detectron2 configs (model + training hyperparameters)
- `custom/` — custom dataset mapper / model components (RGB-D support, regression head, etc.)
- `detectron2/` — Detectron2 source (vendored/submodule-style)

---

## Dataset format

This project assumes a COCO-style dataset with extra fields for depth and diameter.

Typical expectations:

- **RGB image** path in `images[].file_name`
- **Depth map** path stored either:
  - in `images[].depth_file`, or
  - in annotations (implementation-dependent; see `custom/`)
- **Diameter** stored per instance annotation, e.g. `annotations[].diameter_cm`

Example (illustrative):

```json
{
  "images": [
    {
      "id": 3,
      "file_name": "rgb/FRM_0002.png",
      "depth_file": "depth/FRM_0002_Depth.png",
      "width": 1280,
      "height": 720
    }
  ],
  "annotations": [
    {
      "id": 10,
      "image_id": 3,
      "category_id": 1,
      "bbox": [x, y, w, h],
      "segmentation": [...],
      "diameter_cm": 7.42
    }
  ],
  "categories": [{ "id": 1, "name": "apple" }]
}

```
## Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{mazzotti_apple_diameter_estimator,
  author       = {Nicola Mazzotti},
  title        = {apple-diameter-estimator: RGB-D Apple Diameter Estimation with Detectron2},
  year         = {2025},
  howpublished = {\url{https://github.com/Kirin930/apple-diameter-estimator}},
}
