"""
Tomato-only detector: finds tomatoes by colour (HSV), then classifies each crop as
Raw (unripe), Ripe, or Rotten and shows whether it is safe to eat.

Uses produce_database + produce_analysis (same heuristics as the main app).
COCO YOLO does not include tomato, so this file does not run YOLO.

  pip install opencv-python numpy
  python detect_tomato.py          # webcam
  python detect_tomato.py img.jpg  # single image (press any key to close)
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from produce_analysis import estimate_produce_condition
from produce_database import PRODUCE_INFO


KB = "tomato"


def _tomato_colour_mask(hsv: np.ndarray) -> np.ndarray:
    """Ripe red (two hue wraps) + green unripe tomato."""
    info = PRODUCE_INFO[KB]
    r_lo, r_hi = np.array(info["ripe_hsv"][0]), np.array(info["ripe_hsv"][1])
    u_lo, u_hi = np.array(info["unripe_hsv"][0]), np.array(info["unripe_hsv"][1])

    red_core = cv2.inRange(hsv, r_lo, r_hi)
    red_wrap = cv2.inRange(hsv, np.array([170, 120, 90]), np.array([180, 255, 255]))
    green = cv2.inRange(hsv, u_lo, u_hi)
    return red_core | red_wrap | green


def _morph(mask: np.ndarray, k: int = 7) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    return m


def find_tomato_boxes(
    frame_bgr: np.ndarray, min_area_ratio: float = 0.004
) -> list[tuple[int, int, int, int]]:
    h, w = frame_bgr.shape[:2]
    min_area = max(1200, int(h * w * min_area_ratio))
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = _morph(_tomato_colour_mask(hsv), 9)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        x1, y1 = max(0, x - 6), max(0, y - 6)
        x2, y2 = min(w - 1, x + bw + 6), min(h - 1, y + bh + 6)
        boxes.append((x1, y1, x2, y2))
    return _nms_boxes(boxes, iou_thresh=0.35)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    ba = (bx2 - bx1) * (by2 - by1)
    return inter / float(aa + ba - inter + 1e-6)


def _nms_boxes(boxes: list[tuple[int, int, int, int]], iou_thresh: float) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    areas = [((b[2] - b[0]) * (b[3] - b[1]), b) for b in boxes]
    areas.sort(key=lambda t: -t[0])
    kept: list[tuple[int, int, int, int]] = []
    for _area, b in areas:
        if any(_iou(b, k) > iou_thresh for k in kept):
            continue
        kept.append(b)
    return kept


def safety_message(status: str) -> tuple[str, tuple[int, int, int]]:
    """(short headline, BGR colour for box)."""
    if status == "Rotten":
        return "NOT SAFE — do not eat", (0, 0, 255)
    if status == "Ripe":
        return "Safe to eat (wash first)", (0, 200, 80)
    if status == "Unripe / Raw":
        return "Caution — green tomatoes: limit intake", (0, 165, 255)
    if status == "Hard to tell":
        return "Uncertain — inspect before eating", (0, 220, 220)
    return "No tomato detected in this region", (128, 128, 128)


def draw_tomato_panel(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    status: str,
    safety_line: str,
    note: str,
    box_color: tuple[int, int, int],
):
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    panel_y = max(y1 - 118, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, panel_y), (x1 + 320, y1), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    ty = panel_y + 20
    lines = [
        ("TOMATO", (255, 255, 255), 0.55, 1),
        (f"State : {status}", (120, 230, 255), 0.48, 1),
        (f"Eat?  : {safety_line[:38]}", (200, 230, 120) if "Safe" in safety_line else (100, 150, 255), 0.44, 1),
        (f"Tip   : {note[:38]}", (190, 190, 190), 0.38, 1),
    ]
    for text, color, scale, th in lines:
        cv2.putText(frame, text, (x1 + 8, ty), font, scale, color, th, cv2.LINE_AA)
        ty += 24


def annotate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    boxes = find_tomato_boxes(out)
    if not boxes:
        cv2.putText(
            out,
            "No tomato-like colour found — show a red or green tomato",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 180, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    for x1, y1, x2, y2 in boxes:
        roi = out[y1:y2, x1:x2]
        status, _kb_safe, note = estimate_produce_condition(roi, KB)
        safety_line, box_color = safety_message(status)
        draw_tomato_panel(out, x1, y1, x2, y2, status, safety_line, note, box_color)

    cv2.putText(
        out,
        "Tomato-only | Q=quit",
        (10, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    return out


def run_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Tomato detector — webcam. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        vis = annotate_frame(frame)
        cv2.imshow("Tomato detector", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str):
    im = cv2.imread(path)
    if im is None:
        print("Could not read:", path)
        sys.exit(1)
    vis = annotate_frame(im)
    cv2.imshow("Tomato detector", vis)
    print("Press any key to close window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) > 1:
        run_image(sys.argv[1])
    else:
        run_webcam()


if __name__ == "__main__":
    main()
