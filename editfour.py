import cv2
import numpy as np
import time
import threading
import queue

from ultralytics import YOLO
import speech_recognition as sr
import pyttsx3

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# ─────────────────────────────────────────────
# VOICE ENGINE
# ─────────────────────────────────────────────
engine = pyttsx3.init()
engine.setProperty("rate", 150)

tts_queue = queue.Queue()

def speak_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=speak_worker, daemon=True).start()

def speak(text):
    with tts_queue.mutex:
        tts_queue.queue.clear()
    tts_queue.put(text)

# ─────────────────────────────────────────────
# VOICE INPUT
# ─────────────────────────────────────────────
recognizer = sr.Recognizer()
target_object = None

def listen():
    global target_object
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).lower()
            print("You said:", text)

            target_object = text.strip()
            speak(f"Searching for {target_object}")

        except:
            pass

threading.Thread(target=listen, daemon=True).start()

# ─────────────────────────────────────────────
# RIPENESS + ROTTEN DETECTION
# ─────────────────────────────────────────────
def detect_condition(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    brightness = np.mean(hsv[:, :, 2])
    dark_pixels = np.sum(hsv[:, :, 2] < 50)

    if dark_pixels > 500:
        return "Rotten"
    elif brightness > 150:
        return "Ripe"
    else:
        return "Unripe"

# ─────────────────────────────────────────────
# MAIN CAMERA LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)

last_spoken = ""
cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame, conf=0.5)

    found = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            condition = detect_condition(roi)

            # 🎯 TARGET FILTER (MAIN IMPROVEMENT)
            if target_object:
                if target_object not in label:
                    continue

            found = True

            # DRAW
            color = (0,255,0) if condition=="Ripe" else (0,255,255)
            if condition == "Rotten":
                color = (0,0,255)

            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)

            text = f"{label} ({condition}) {conf:.2f}"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 🔊 SPEAK ONCE
            key = f"{label}-{condition}"
            if key != last_spoken and time.time() > cooldown:
                speak(f"{label} detected. It is {condition}")
                last_spoken = key
                cooldown = time.time() + 5

    # ❌ TARGET NOT FOUND
    if target_object and not found:
        cv2.putText(frame, f"Searching for: {target_object}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,200,255), 2)

    cv2.imshow("Smart Fruit & Vegetable Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)