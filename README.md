# Multispectral-Pedestrian-Segmentation-Ensemble-CMP486

## Short Description

A cascaded computer vision pipeline that detects and segments pedestrians in challenging visibility conditions using paired infrared and visible images from the LLVIP dataset. YOLOv11s detects pedestrians on fused IR+visible images, and SAM 2 refines each detection into a pixel-level instance mask. A veto mechanism discards masks that fail geometric validation, improving ensemble precision over the YOLO-only baseline.

This project was completed by American University of Sharjah students for the course CMP486: Computer Vision under the guidance of Dr. Omar Arif.

## Project Overview

Pedestrian detection degrades in extreme environments. In very hot climates, thermal cameras lose contrast as roads and buildings approach human body temperature. Visible cameras fail under glare, haze, shadows, and low light. Multispectral fusion combines both sensor types into a single input, making detection robust where either sensor alone would fail.

This project uses multispectral fusion to combine information from two sensor types:

- Infrared images capture thermal information.
- Visible images capture color, texture, and shape information.

The final pipeline is:

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
