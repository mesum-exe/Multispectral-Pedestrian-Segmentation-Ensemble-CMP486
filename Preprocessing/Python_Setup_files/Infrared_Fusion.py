def create_fused_image(ir_path: Path, vis_path: Path, res: int = TARGET_RES) -> np.ndarray:
    """Create the saved YOLO input: IR plus visible green and blue channels."""
    ir_gray = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    vis_bgr = cv2.imread(str(vis_path))
    if ir_gray is None or vis_bgr is None:
        raise FileNotFoundError(f"Could not read pair: {ir_path.name}")

    ir_resized = cv2.resize(ir_gray, (res, res), interpolation=cv2.INTER_AREA)
    vis_resized = cv2.resize(vis_bgr, (res, res), interpolation=cv2.INTER_AREA)
    return np.stack([ir_resized, vis_resized[:, :, 1], vis_resized[:, :, 0]], axis=2).astype(np.uint8)
