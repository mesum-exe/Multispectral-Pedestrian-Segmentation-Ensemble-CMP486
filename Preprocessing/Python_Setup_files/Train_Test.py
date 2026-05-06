TARGET_RES = 1024
CLASS_ID = 0
TRAIN_RATIO = 0.85

for split in ["train", "val"]:
    for subfolder in ["images", "labels"]:
        (processed_dataset_dir / split / subfolder).mkdir(parents=True, exist_ok=True)


paired_stems = [p.stem for p in all_ir_train
                if (llvip_visible_train_dir / (p.stem + ".jpg")).exists()]

random.seed(42)
random.shuffle(paired_stems)
n_train = int(len(paired_stems) * TRAIN_RATIO)
split_stems = {"train": paired_stems[:n_train], "val": paired_stems[n_train:]}

print(f"Total pairs: {len(paired_stems)}")
print(f"Training split: {len(split_stems['train'])}")
print(f"Validation split: {len(split_stems['val'])}")

expected_train = len(split_stems["train"])
expected_val = len(split_stems["val"])
current_train_images = len(list((processed_dataset_dir / "train" / "images").glob("*.jpg")))
current_train_labels = len(list((processed_dataset_dir / "train" / "labels").glob("*.txt")))
current_val_images = len(list((processed_dataset_dir / "val" / "images").glob("*.jpg")))
current_val_labels = len(list((processed_dataset_dir / "val" / "labels").glob("*.txt")))

preprocessing_complete = (
    current_train_images >= expected_train and current_train_labels >= expected_train and
    current_val_images >= expected_val and current_val_labels >= expected_val
)

if preprocessing_complete:
    print("Processed dataset already looks complete. Skipping image and label creation.")
else:
    print("Creating missing fused images and YOLO labels.")
    total_imgs = 0
    total_boxes = 0

    for split, stems in split_stems.items():
        split_img_dir = processed_dataset_dir / split / "images"
        split_label_dir = processed_dataset_dir / split / "labels"

        for stem in stems:
            image_out = split_img_dir / f"{stem}.jpg"
            label_out = split_label_dir / f"{stem}.txt"
            if image_out.exists() and label_out.exists():
                continue

            ir_src = llvip_infrared_train_dir / f"{stem}.jpg"
            vis_src = llvip_visible_train_dir / f"{stem}.jpg"
            xml_src = llvip_annotations_dir / f"{stem}.xml"
            if not xml_src.exists():
                continue

            fused = create_fused_image(ir_src, vis_src, TARGET_RES)
            cv2.imwrite(str(image_out), fused)
            boxes = convert_voc_xml_to_yolo(xml_src, label_out, TARGET_RES, TARGET_RES)
            total_imgs += 1
            total_boxes += boxes

        print(f"{split}: checked {len(stems)} images")

    print(f"Images created or repaired: {total_imgs}")
    print(f"Boxes written: {total_boxes}")

# Creating data.yaml to be used downstream
data_yaml_path = processed_dataset_dir / "data.yaml"
data_yaml = {
    "path": str(processed_dataset_dir),
    "train": "train/images",
    "val": "val/images",
    "nc": 1,
    "names": ["Pedestrian"],
}

with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"data.yaml saved: {data_yaml_path}")
