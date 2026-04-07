from ultralytics import YOLO
import cv2

# Load pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# List of common fruits & vegetables (can expand)
produce_classes = [
    "banana", "apple", "orange", "carrot", "broccoli",
    "tomato", "potato", "cucumber"
]

def get_status(label):
    # Placeholder logic (you can train later)
    return "Fresh"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in produce_classes:
                status = get_status(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} - {status}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

    cv2.imshow("AI Fruit & Vegetable Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()