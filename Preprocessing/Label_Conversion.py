def convert_voc_xml_to_yolo(xml_path: Path, label_path: Path, fallback_w: int, fallback_h: int) -> int:
    """Convert one LLVIP XML annotation into one YOLO label file."""
    root = ElementTree.parse(str(xml_path)).getroot()

    size_node = root.find("size")
    if size_node is not None and size_node.find("width") is not None and size_node.find("height") is not None:
        image_w = int(float(size_node.find("width").text))
        image_h = int(float(size_node.find("height").text))
    else:
        image_w, image_h = fallback_w, fallback_h

    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        cx = ((xmin + xmax) / 2) / image_w
        cy = ((ymin + ymax) / 2) / image_h
        bw = (xmax - xmin) / image_w
        bh = (ymax - ymin) / image_h
        cx, cy, bw, bh = [min(max(v, 0.0), 1.0) for v in (cx, cy, bw, bh)]
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    label_path.write_text("\n".join(lines))
    return len(lines)
