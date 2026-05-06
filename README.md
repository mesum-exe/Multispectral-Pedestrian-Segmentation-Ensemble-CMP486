# Multispectral-Pedestrian-Segmentation-Ensemble-CMP486

## Short Description

A cascaded computer vision pipeline that detects and segments pedestrians in challenging visibility conditions using paired infrared and visible images from the LLVIP dataset. YOLOv11s detects pedestrians on fused IR+visible images, and SAM 2 refines each detection into a pixel-level instance mask. A veto mechanism discards masks that fail geometric validation, improving ensemble precision over the YOLO-only baseline.

This project was completed by American University of Sharjah students for the course CMP486: Computer Vision under the guidance of Dr. Omar Arif.

## Project Overview

Pedestrian detection degrades in extreme environments. In very hot climates, thermal cameras lose contrast as roads and buildings approach human body temperature. Visible cameras fail under glare, haze, shadows, and low light. Multispectral fusion combines both sensor types into a single input, making detection robust where either sensor alone would fail.

This project uses multispectral fusion to combine information from two sensor types:

- Infrared images capture thermal information.
- Visible images capture color, texture, and shape information.

### The final pipeline is:

```text
LLVIP infrared + visible image pair
        │
        v
  Fused 3-channel image
  (IR → R channel, Visible G → G, Visible B → B)
        │
        v
  YOLOv11s pedestrian detection
  (trained on fused images at 1024×1024)
        │
        v
  SAM 2 segmentation
  (runs on visible RGB, prompted by YOLO boxes)
        │
        v
  Mask veto logic
  (rejects empty, oversized, or spatially misaligned masks)
        │
        v
  Final pedestrian boxes, masks, metrics, and plots
```

## Setup - LLVIP Dataset
The LLVIP (Low-Light Visible-Infrared Paired) dataset provides pixel-aligned infrared and visible image pairs captured in low-light surveillance scenes.

Source: https://bupt-ai-cz.github.io/LLVIP/
Size: ~15,488 paired images (infrared + visible), ~4 GB
Annotations: Pascal VOC XML format, pedestrian class only
Split used: LLVIP training split (12,025 pairs after alignment check), divided 85/15 into train/val

### How to download:

1. Visit: https://bupt-ai-cz.github.io/LLVIP/
2. Follow the instructions from the provided link to access the dataset. You will need to enter your information.
3. Download LLVIP.zip from the provided link.
4. Place the zip at:
```
   /home/vteam5/multispectral_pedestrian_ensemble/LLVIP.zip
```
5. The Preprocessing notebook will extract and verify the archive automatically when it is ran.

### Expected folder structure after extraction:
```
llvip_dataset/
├── infrared/
│   └── train/       <-- infrared .jpg images
├── visible/
│   └── train/       <-- visible .jpg images
└── Annotations/     <-- flat folder of Pascal VOC XML files (no train/ subfolder)
```
## Repository Structure
```
Multispectral-Pedestrian-Segmentation-Ensemble-CMP486/
├── Preprocessing/
├── YOLO_Detection/
├── Ensemble_Pipeline/
├── Evaluation_Results/
└── README.md
```
```Preprocessing/```
- Contains the data preparation pipeline.
- Reads paired IR and visible images from LLVIP
- Converts Pascal VOC XML annotations to YOLO format
- Creates fused 3-channel images (IR + visible green + visible blue channels)
- Splits data 85/15 into train and val sets
- Writes data.yaml for YOLO training.

**Output:**
processed_dataset/
├── train/
│   ├── images/     ← fused .jpg images (1024×1024)
│   └── labels/     ← YOLO format .txt annotations
├── val/
│   ├── images/
│   └── labels/
└── data.yaml

```
YOLO_Detection/
```
- Contains the YOLOv11s training and evaluation notebook.
- Loads the trained checkpoint (or trains from scratch if none exists).
- Runs a confidence threshold sweep across [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50] to identify the optimal threshold for YOLO
- Runs official model.val() for mAP metrics.
- Exports: Qualitative prediction plots, yolo_detections.json — a bridge file of YOLO boxes in xyxy absolute pixel format for the SAM 2 stage.

**Key results (through our training run):**
mAP@50: 0.8805
mAP@50-95: 0.4449
Precision: 0.9268
Recall: 0.8877
Recommended confidence threshold: 0.30 (from F1 sweep)

``` Ensemble_Pipeline/ ```
- Contains the full cascaded ensemble notebook (Cascaded_MultiSpectral_Pedestrian_Segmentation_Best.ipynb).
- Loads the YOLO checkpoint and SAM 2 Hiera-Large.
- Runs YOLO on fused images to get bounding boxes
- Passes those boxes to SAM 2 on the original visible RGB image
- Applies a geometric veto that discards masks that are empty, too small, too large, or **misaligned with the YOLO box** (main factor).
- Evaluates YOLO-only vs ensemble side by side.
- Results are cached to disk after the ensemble run to survive runtime disconnects.
  
**Veto parameters:**
min_area_ratio: 0.05 — masks smaller than 5% of the YOLO box area are rejected
max_area_ratio: 1.25 — masks larger than 125% of the YOLO box area are rejected
min_box_iou: 0.30 — masks whose bounding box overlaps the YOLO box by less than 30% IoU are rejected

``` Evaluation_Results/ ```
**Contains saved output artefacts from both the YOLO and ensemble stages, including:**
1. ```section4_alignment_check.png``` --> IR / visible / fused overlay for 3 sample pairs
2. ```training_curves.png``` --> box loss, classification loss, mAP over training epochs
3. ```confidence_sweep.csv``` and ```confidence_sweep.png``` --> threshold sweep results across 8 confidence values
4. ```yolo_baseline_metrics.json``` --> official mAP, precision, and recall scores
5. ```qualitative_samples.png``` --> YOLO prediction vs ground truth visualisation
6.```yolo_detections.json``` --> YOLO bridge export (xyxy boxes) for SAM 2
7. ```veto_settings.json``` --> veto parameters used for the ensemble run
8. ```ensemble_results_cache_conf_*.json``` --> cached ensemble results per confidence threshold
9. ```ensemble_metrics.json``` --> aggregated YOLO vs ensemble evaluation metrics
10. ```ensemble_per_image_metrics.csv``` --> per-image breakdown of TP, FP, FN for both stages
11. ```ensemble_qualitative.png``` --> 4-column visualisation: IR / visible / YOLO boxes / SAM 2 masks
12. ```ensemble_bridge_results.json``` --> final masks and boxes for downstream use
