# ============================================================
# detector.py — Core inference + Grad-CAM logic
# Used by app.py (Streamlit frontend)
# ============================================================

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Class definitions ─────────────────────────────────────────
CLASS_NAMES = ["scratch", "dent", "broken_part", "paint_damage"]

# Color per class (BGR for OpenCV)
CLASS_COLORS = {
    "scratch":      (0,   200, 255),   # cyan
    "dent":         (0,   100, 255),   # orange
    "broken_part":  (0,   0,   220),   # red
    "paint_damage": (180, 105, 255),   # pink
}

# Cosmetic vs Functional
FUNCTIONAL_CLASSES = {"broken_part"}


def load_model(model_path: str) -> YOLO:
    """Load YOLOv8 model from .pt file."""
    model = YOLO(model_path)
    return model


def run_detection(model: YOLO, image: Image.Image, conf: float = 0.25, iou: float = 0.6):
    """
    Run YOLOv8 detection on a PIL image.
    Returns annotated image (PIL) + structured damage report (dict).
    """
    # Convert PIL → numpy (RGB)
    img_np = np.array(image.convert("RGB"))

    # Run inference
    results = model.predict(
        source=img_np,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False,
    )[0]

    # Draw bounding boxes on a copy
    annotated = img_np.copy()
    detections = []

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        # Severity from bounding box area ratio
        box_area  = (x2 - x1) * (y2 - y1)
        img_area  = annotated.shape[0] * annotated.shape[1]
        area_ratio = box_area / max(img_area, 1)

        if area_ratio < 0.02:
            severity = "Low"
        elif area_ratio < 0.08:
            severity = "Medium"
        else:
            severity = "High"

        impact = "Functional" if cls_name in FUNCTIONAL_CLASSES else "Cosmetic"

        # Draw box
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        color_rgb = (color[2], color[1], color[0])  # BGR→RGB
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 3)

        # Label background
        label = f"{cls_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color_rgb, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections.append({
            "damage_type":      cls_name,
            "bounding_box":     [x1, y1, x2, y2],
            "severity":         severity,
            "impact":           impact,
            "confidence_score": round(confidence, 4),
            "area_ratio":       round(area_ratio, 4),
        })

    # Sort by confidence descending
    detections.sort(key=lambda x: x["confidence_score"], reverse=True)

    report = {
        "total_damages_detected": len(detections),
        "damages": detections,
        "high_risk_flag": any(
            d["severity"] == "High" or d["confidence_score"] < 0.4
            for d in detections
        ),
    }

    return Image.fromarray(annotated), report


def generate_gradcam(model: YOLO, image: Image.Image) -> Image.Image:
    """
    Generate a Grad-CAM style heatmap overlay using the
    last conv layer feature maps from YOLOv8 backbone.
    Falls back to a gradient-free activation map if hooks fail.
    """
    img_np  = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_np, (640, 640))

    # Normalize for torch
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)

    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    # Hook into the last Conv layer of YOLOv8 backbone
    hook = None
    try:
        # Navigate to last backbone conv layer
        backbone_layers = list(model.model.model.children())
        last_conv = None
        for layer in reversed(backbone_layers):
            if hasattr(layer, 'conv'):
                last_conv = layer.conv
                break
            if isinstance(layer, torch.nn.Conv2d):
                last_conv = layer
                break

        if last_conv is not None:
            hook = last_conv.register_forward_hook(hook_fn)

        with torch.no_grad():
            model.model(tensor)

        if hook:
            hook.remove()

    except Exception:
        if hook:
            try:
                hook.remove()
            except Exception:
                pass

    # Build heatmap from activations
    if activations:
        feat = activations[0].squeeze(0)           # (C, H, W)
        heatmap = feat.mean(dim=0).numpy()         # average over channels
    else:
        # Fallback: use raw image luminance as proxy
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
        heatmap = cv2.GaussianBlur(gray, (31, 31), 0)

    # Normalise to 0-1
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # Resize heatmap to original image size
    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap and blend
    colored = cm.jet(heatmap_resized)[:, :, :3]   # RGB, drop alpha
    colored = (colored * 255).astype(np.uint8)

    blended = cv2.addWeighted(img_np, 0.55, colored, 0.45, 0)

    return Image.fromarray(blended)


def severity_score(damages: list) -> float:
    """
    Compute overall severity score 0–100 from list of detections.
    Used for the gauge chart.
    """
    if not damages:
        return 0.0

    severity_map = {"Low": 20, "Medium": 55, "High": 90}
    scores = [severity_map.get(d["severity"], 0) * d["confidence_score"]
              for d in damages]

    # Weighted average, capped at 100
    return min(100.0, sum(scores) / len(scores))
