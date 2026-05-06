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
---
## Important Note on Paths:

Multiple folders have been created to imitate the paths in the Notebooks. However, this project was completed on a University-provided virtual machine, so paths will need to be adjusted. We will fix this in the future.

The Path Configuration can be separately changed in ```Preprocessing\ --> Project_Setup.ipnyb```. These changes can easily be applied to the main notebook ```Ensemble_Pipeline\ --> Cascaded_RGB-T_Segmentation_Best.ipnyb``` as all sections are well formatted for clarity.

```results\``` and ```model_checkpoints``` directories have been duplicated from ```SAM2\``` and ```YOLO_Detection\``` into ```Ensemble_Artefacts\``` to keep SAM2 and YOLOv11s download paths active whether the isolated notebooks are ran, or the complete notebook is ran.

The root ```/home/vteam5/``` must also be replaced to the name of this repository in all paths. 

Recommended Structure:
```
Multispectral-Pedestrian-Segmentation-Ensemble-CMP486/  <-- Root (replaces /home/vteam5/...)
│
├── Preprocessing/
│   └── [Alignment scripts and sample overlays]
├── YOLO/
│   ├── results/
│   │   └── pedestrian_multispectral_yolo11s/
│   │       ├── weights/
│   │       │   └── best.pt
│   │       ├── args.yaml
│   │       └── results.csv
│   └── training_scripts/
├── SAM2/
│   ├── model_checkpoints/
│   │   └── sam2_hiera_large.pt             
│   ├── sam2/                               
│   ├── sam2_configs/
│   └── setup.py
├── Ensemble_Pipeline/
│   ├── Cascaded_MultiSpectral_Pedestrian_Segmentation_Best.ipynb
│   └── ensemble_bridge_results.json
└── Evaluation_Results/
    ├── ensemble_metrics.json
    ├── ensemble_per_image_metrics.csv
    └── ensemble_qualitative.png
```
All instructions ahead will assume the same file structure as ours is being used.

---
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
4. Place the zip into these nested folders:
```
   /home/vteam5/multispectral_pedestrian_ensemble/LLVIP.zip
```
or change all mentions of ```/home/vteam5/multispectral_pedestrian_ensemble/``` to ```Multispectral-Pedestrian-Segmentation-Ensemble-CMP486```.

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
---
## Repository Structure
```
Multispectral-Pedestrian-Segmentation-Ensemble-CMP486/
├── Preprocessing/
├── YOLO_Detection/
├── SAM2/
├── Ensemble_Pipeline/
├── Evaluation_Results/
└── README.md
```
---
```Preprocessing/```
- Contains the data preparation pipeline.
- Reads paired IR and visible images from LLVIP
- Converts Pascal VOC XML annotations to YOLO format
- Creates fused 3-channel images (IR + visible green + visible blue channels)
- Splits data 85/15 into train and val sets
- Writes data.yaml for YOLO training.

**Output:**
```
processed_dataset/
├── train/
│   ├── images/     <-- fused .jpg images (1024×1024)
│   └── labels/     <-- YOLO format .txt annotations
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```
---
```YOLO_Detection/```
- Contains the YOLOv11s training and evaluation notebook.
- Loads the trained checkpoint (or trains from scratch if none exists).
- Runs a confidence threshold sweep across [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50] to identify the optimal threshold for YOLO
- Runs official model.val() for mAP metrics.
- Exports: Qualitative prediction plots, yolo_detections.json --> a bridge file of YOLO boxes in xyxy absolute pixel format for the SAM 2 stage.

**Key results (through our training run):**
1. ```mAP@50```: 0.8805    --> mean Average Precision calculated over a 50% minimum IoU
2. ```mAP@50-95```: 0.4449 --> mean Average Precision calculated over a 50 to 95% minimum IoU
3. ```Precision```: 0.9268
4. ```Recall```: 0.8877
Recommended confidence threshold: 0.30 (from F1 sweep)
---
```SAM2/``` 
- Logic for receiving (xyxy) coordinates from YOLOv11 and generating localized masks on high-resolution visible RGB imagery.Stability Scoring 
- Outputs a predicted intersection-over-union (IoU) score for each mask, used to filter out low-confidence segments.Zero-Shot Transfer 
- The capability to segment pedestrians without additional fine-tuning, leveraging SAM 2's foundational training.

**Veto parameters:**
1. ```min_area_ratio```: 0.05 --> masks smaller than 5% of the YOLO box area are rejected
2. ```max_area_ratio```: 1.25 --> masks larger than 125% of the YOLO box area are rejected
3. ```min_box_iou```: 0.30 --> masks whose bounding box overlaps the YOLO box by less than 30% IoU are rejected

---
``` Ensemble_Pipeline/ ```
- Contains the full cascaded ensemble notebook (Cascaded_MultiSpectral_Pedestrian_Segmentation_Best.ipynb).
- Loads the YOLO checkpoint and SAM 2 Hiera-Large.
- Runs YOLO on fused images to get bounding boxes
- Passes those boxes to SAM 2 on the original visible RGB image
- Applies a geometric veto that discards masks that are empty, too small, too large, or **misaligned with the YOLO box** (main factor).
- Evaluates YOLO-only vs ensemble side by side.
- Results are cached to disk after the ensemble run to survive runtime disconnects.
- ```ensemble_bridge_results.json``` --> Serialized output containing the final coordinates and mask data for all verified pedestrian detections.
  
---
``` Evaluation_Results/ ```

**Contains saved output artefacts from the ensemble stage, including:**
1. ```ensemble_metrics.json``` --> Aggregated performance data, including final Precision, Recall, and mAP scores for the combined YOLOv11 and SAM 2 pipeline.
2. ```ensemble_per_image_metrics.csv``` --> A detailed breakdown of True Positives, False Positives, and False Negatives for every image in the test set.
3. ```ensemble_qualitative.png``` --> A visual comparison showing original IR/Visible frames alongside the final pedestrian detections and SAM 2 segmentations.

## Models

| Model | Role | Checkpoint |
|----------|:-------------:|------:|
| Fine-tuned YOLOv11s | Pedestrian detection on fused images | ```results/pedestrian_multispectral_yolo11s/weights/best.pt``` (in files) --> ```results/weights/best.pt ``` (adjusted for repository)| 
| SAM 2 Hiera-Large | Instance segmentation on visible RGB | ```model_checkpoints/sam2_hiera_large.pt``` |
| SAM 2 Hiera-Large | Instance segmentation on visible RGB | ```model_checkpoints/sam2_hiera_large.pt``` |

SAM 2 weights are downloaded automatically on first run from:

https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

Ensure this is downloaded into the provided ```model_checkpoints\```

## Environment 

Python 3.12
PyTorch 2.6.0+cu124
Ultralytics 8.4.45
CUDA 12.2 — NVIDIA Quadro RTX 4000 (8 GB VRAM)
Conda environment: vision_proj_v3

## Install dependencies:
```
bashconda create -n vision_proj_v3 python=3.12 -y
conda activate vision_proj_v3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics pyyaml opencv-python-headless matplotlib pandas scipy
git clone https://github.com/facebookresearch/sam2.git /home/vteam5/sam2
pip install -e /home/vteam5/sam2
```
