import cv2
import numpy as np
from ultralytics import YOLO

# pip install ultralytics opencv-python

# Load pretrained YOLOv8 model (trained on COCO - includes some fruits/vegetables)
model = YOLO("yolov8n.pt")  # Downloads automatically on first run

# --- Knowledge base for produce ---
PRODUCE_INFO = {
    "apple": {
        "type": "fruit",
        "ripe_colors_hsv": {"lower": [0, 100, 100], "upper": [15, 255, 255]},   # red
        "unripe_colors_hsv": {"lower": [35, 80, 80], "upper": [85, 255, 255]},  # green
        "safe": True,
        "notes": "Safe raw or cooked. Avoid seeds (mild toxin).",
    },
    "banana": {
        "type": "fruit",
        "ripe_colors_hsv": {"lower": [20, 100, 150], "upper": [35, 255, 255]},  # yellow
        "unripe_colors_hsv": {"lower": [35, 80, 80], "upper": [85, 255, 255]},  # green
        "safe": True,
        "notes": "Safe. Unripe may cause bloating.",
    },
    "orange": {
        "type": "fruit",
        "ripe_colors_hsv": {"lower": [10, 150, 150], "upper": [25, 255, 255]},
        "unripe_colors_hsv": {"lower": [35, 80, 80], "upper": [85, 255, 255]},
        "safe": True,
        "notes": "Safe raw. Rich in vitamin C.",
    },
    "broccoli": {
        "type": "vegetable",
        "ripe_colors_hsv": {"lower": [35, 80, 50], "upper": [85, 255, 200]},    # deep green = mature
        "unripe_colors_hsv": {"lower": [35, 60, 150], "upper": [85, 200, 255]}, # bright green = young
        "safe": True,
        "notes": "Safe raw or cooked. Cooking improves digestibility.",
    },
    "carrot": {
        "type": "vegetable",
        "ripe_colors_hsv": {"lower": [8, 150, 150], "upper": [20, 255, 255]},   # orange
        "unripe_colors_hsv": {"lower": [8, 80, 200], "upper": [20, 180, 255]},  # pale orange
        "safe": True,
        "notes": "Safe raw or cooked.",
    },
    "tomato": {
        "type": "fruit",  # botanically a fruit
        "ripe_colors_hsv": {"lower": [0, 150, 100], "upper": [10, 255, 255]},   # red
        "unripe_colors_hsv": {"lower": [35, 80, 80], "upper": [85, 255, 255]},  # green
        "safe": True,
        "notes": "Safe when ripe. Green tomatoes contain solanine - eat in moderation.",
    },
    "potato": {
        "type": "vegetable",
        "ripe_colors_hsv": {"lower": [15, 30, 100], "upper": [30, 120, 200]},
        "unripe_colors_hsv": {"lower": [35, 40, 80], "upper": [85, 150, 180]},
        "safe": True,
        "notes": "Must be cooked. Raw potato is hard to digest. Green parts contain solanine - AVOID.",
    },
    "corn": {
        "type": "vegetable",
        "ripe_colors_hsv": {"lower": [20, 100, 150], "upper": [35, 255, 255]},  # yellow
        "unripe_colors_hsv": {"lower": [35, 60, 180], "upper": [85, 180, 255]}, # pale green/yellow
        "safe": True,
        "notes": "Safe raw or cooked.",
    },
}

# COCO class names that are produce (YOLOv8 default COCO classes)
COCO_PRODUCE = {
    "apple", "banana", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut",
    "cake", "sandwich"
}

# Map COCO names to our knowledge base keys
COCO_TO_KB = {
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
}


def estimate_ripeness(roi_bgr, produce_name):
    """Estimate ripeness by checking dominant color in the bounding box ROI."""
    if produce_name not in PRODUCE_INFO:
        return "Unknown"

    info = PRODUCE_INFO[produce_name]
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    ripe_lower = np.array(info["ripe_colors_hsv"]["lower"])
    ripe_upper = np.array(info["ripe_colors_hsv"]["upper"])
    unripe_lower = np.array(info["unripe_colors_hsv"]["lower"])
    unripe_upper = np.array(info["unripe_colors_hsv"]["upper"])

    ripe_pixels = np.sum(cv2.inRange(hsv_roi, ripe_lower, ripe_upper) > 0)
    unripe_pixels = np.sum(cv2.inRange(hsv_roi, unripe_lower, unripe_upper) > 0)

    if ripe_pixels > unripe_pixels and ripe_pixels > 500:
        return "Ripe"
    elif unripe_pixels > 500:
        return "Unripe / Raw"
    else:
        return "Hard to tell"


def get_size_label(box_area, frame_area):
    """Estimate relative size of the object."""
    ratio = box_area / frame_area
    if ratio > 0.3:
        return "Large"
    elif ratio > 0.10:
        return "Medium"
    else:
        return "Small"


def draw_info_box(frame, x1, y1, x2, y2, label, ripeness, size, safe, notes, obj_type):
    """Draw a clean info overlay."""
    # Box color: green for safe, red for unsafe
    box_color = (0, 200, 80) if safe else (0, 60, 220)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Background panel for text
    panel_y = max(y1 - 120, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, panel_y), (x1 + 280, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tx, ty = x1 + 6, panel_y + 18

    safe_text = "SAFE TO EAT" if safe else "CAUTION"
    safe_color = (80, 255, 120) if safe else (80, 120, 255)

    lines = [
        (f"{label.upper()} ({obj_type})", (255, 255, 255), 0.52, 1),
        (f"Size    : {size}",             (200, 200, 200), 0.45, 1),
        (f"Status  : {ripeness}",         (100, 220, 255), 0.45, 1),
        (f"Safety  : {safe_text}",        safe_color,      0.45, 1),
        (f"Note    : {notes[:34]}",       (180, 180, 180), 0.38, 1),
    ]

    for text, color, scale, thick in lines:
        cv2.putText(frame, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
        ty += 22


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting Fruit & Vegetable Detector...")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_area = frame.shape[0] * frame.shape[1]

    # Run YOLO detection
    results = model(frame, conf=0.45, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Only process produce classes
            if label not in COCO_TO_KB:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop ROI safely
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                continue

            kb_name = COCO_TO_KB[label]
            info = PRODUCE_INFO.get(kb_name, {})

            ripeness = estimate_ripeness(roi, kb_name)
            box_area = (x2 - x1) * (y2 - y1)
            size = get_size_label(box_area, frame_area)

            draw_info_box(
                frame,
                x1, y1, x2, y2,
                label=label,
                ripeness=ripeness,
                size=size,
                safe=info.get("safe", True),
                notes=info.get("notes", "No info available."),
                obj_type=info.get("type", "produce"),
            )

            # Confidence badge
            cv2.putText(frame, f"{conf:.0%}", (x2 - 45, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 180), 1)

    # FPS display
    cv2.putText(frame, "Fruit & Veg Detector | press Q to quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Fruit & Vegetable Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()