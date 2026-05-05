# Multispectral-Pedestrian-Segmentation-Ensemble-CMP486

## Short Description

This project detects and segments pedestrians in challenging heat and low-visibility conditions using paired infrared and visible images from the LLVIP dataset. The system first uses YOLOv11 on fused infrared-visible images for pedestrian detection, then uses SAM 2 to refine detections into instance masks. A conservative validation step checks whether each SAM mask is reasonable before keeping it in the final ensemble output.

## Project Overview

Pedestrian detection can become unreliable in extreme climates. In very hot environments, thermal cameras may lose contrast because roads and buildings can become as hot as, or hotter than, people. Visible cameras can also struggle because of glare, haze, shadows, and low-light conditions.

This project uses multispectral fusion to combine information from two sensor types:

- Infrared images capture thermal information.
- Visible images capture color, texture, and shape information.

The final pipeline is:

```text
LLVIP infrared + visible image pair
        |
        v
Fused 3-channel YOLO input
        |
        v
YOLOv11 pedestrian detection
        |
        v
SAM 2 segmentation on visible RGB image
        |
        v
Mask validation / veto logic
        |
        v
Final pedestrian boxes, masks, metrics, and plots
