val_fused_paths = sorted((processed_dataset_dir / "val" / "images").glob("*.jpg"))
val_ir_paths = [llvip_infrared_train_dir / p.name for p in val_fused_paths]
val_vis_paths = [llvip_visible_train_dir / p.name for p in val_fused_paths]

print(f"Validation images for ensemble: {len(val_fused_paths)}")


ensemble_yolo = YOLO(str(best_checkpoint_path))
all_results = []

for i, (fp, ip, vp) in enumerate(zip(val_fused_paths, val_ir_paths, val_vis_paths)):
    result = run_ensemble(fp, ip, vp, ensemble_yolo, conf=cascade_conf)
    all_results.append(result)
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(val_fused_paths)} images processed")

print(f"Ensemble finished on {len(all_results)} images.")


# Anoether heavy cell. Save in case runtime disconnects.

import json
import numpy as np
from pathlib import Path

# Ensure directory exists
ensemble_results_dir.mkdir(parents=True, exist_ok=True)
ensemble_results_cache_path = ensemble_results_dir / f"ensemble_results_cache_conf_{cascade_conf:.2f}.json"

ensemble_results_cache = {
    "metadata": {
        "cascade_conf": float(cascade_conf),
        "min_area_ratio": float(SAM_MASK_MIN_AREA_RATIO),
        "max_area_ratio": float(SAM_MASK_MAX_AREA_RATIO),
        "min_box_iou": float(SAM_MASK_MIN_BOX_IOU),
        "num_images": len(all_results),
        "image_size": int(TARGET_RES),
        "checkpoint_path": str(best_checkpoint_path),
    },
    "results": []
}

for res in all_results:
    # use .get() with empty lists as fallbacks to prevent the KeyError 
    # if a specific image had zero detections.
    ensemble_results_cache["results"].append({
        "image_name": res["image_name"],
        "yolo_boxes_xyxy": np.asarray(res.get("yolo_boxes_xyxy", [])).tolist(),
        "yolo_conf": np.asarray(res.get("yolo_conf", [])).tolist(),
        "final_boxes_xyxy": np.asarray(res.get("final_boxes_xyxy", [])).tolist(),
        "final_mask_boxes_xyxy": np.asarray(res.get("final_mask_boxes_xyxy", [])).tolist(),
        "final_scores": np.asarray(res.get("final_scores", [])).tolist(),
        "vetoed": int(res.get("vetoed", 0)),
        "veto_reasons": res.get("veto_reasons", []),
    })

with open(ensemble_results_cache_path, "w") as f:
    json.dump(ensemble_results_cache, f, indent=2)

print(f"Successfully saved ensemble results: {ensemble_results_cache_path}")


# run only when runtime disconnects

import json
import numpy as np
import os

ensemble_results_cache_path = ensemble_results_dir / f"ensemble_results_cache_conf_{cascade_conf:.2f}.json"

if os.path.exists(ensemble_results_cache_path):
    print(f" Loading cache from {ensemble_results_cache_path}...")
    
    with open(ensemble_results_cache_path, "r") as f:
        cache_data = json.load(f)
    
    all_results = []
    for item in cache_data["results"]:
        all_results.append({
            "image_name": item["image_name"],
            "yolo_boxes_xyxy": np.array(item["yolo_boxes_xyxy"]),
            "yolo_conf": np.array(item["yolo_conf"]),
            "final_boxes_xyxy": np.array(item["final_boxes_xyxy"]),
            "final_mask_boxes_xyxy": np.array(item["final_mask_boxes_xyxy"]),
            "final_scores": np.array(item["final_scores"]),
            "vetoed": bool(item["vetoed"]),
            "veto_reasons": item["veto_reasons"]
        })
    
    # Sync metadata
    cascade_conf = cache_data["metadata"]["cascade_conf"]
    print(f" Session restored. {len(all_results)} images loaded.")
else:
    print(f" No cache file found at {ensemble_results_cache_path}.")