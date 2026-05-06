# Multispectral-Pedestrian-Segmentation-Ensemble-CMP486

## Short Description

A cascaded computer vision pipeline that detects and segments pedestrians in challenging visibility conditions using paired infrared and visible images from the LLVIP dataset. YOLOv11s detects pedestrians on fused IR+visible images, and SAM 2 refines each detection into a pixel-level instance mask. A veto mechanism discards masks that fail geometric validation, improving ensemble precision over the YOLO-only baseline.

This project was completed by American University of Sharjah students for the course CMP486: Computer Vision under the guidance of Dr. Omar Arif.

## Project Overview

Pedestrian detection degrades in extreme environments. In very hot climates, thermal cameras lose contrast as roads and buildings approach human body temperature. Visible cameras fail under glare, haze, shadows, and low light. Multispectral fusion combines both sensor types into a single input, making detection robust where either sensor alone would fail.

This project uses multispectral fusion to combine information from two sensor types:

- Infrared images capture thermal information.
- Visible images capture color, texture, and shape information.

# The final pipeline is:

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

# How to download:

1. Visit: https://bupt-ai-cz.github.io/LLVIP/
2. Follow the instructions from the provided link to access the dataset. You will need to enter your information.
3. Download LLVIP.zip from the provided link.
4. Place the zip at:
```
   /home/vteam5/multispectral_pedestrian_ensemble/LLVIP.zip
```
5. The Preprocessing notebook will extract and verify the archive automatically when it is ran.

# Expected folder structure after extraction:
```
llvip_dataset/
├── infrared/
│   └── train/       <-- infrared .jpg images
├── visible/
│   └── train/       <-- visible .jpg images
└── Annotations/     <-- flat folder of Pascal VOC XML files (no train/ subfolder)
```
