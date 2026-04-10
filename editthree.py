"""
FRUIT & VEGETABLE DETECTOR  —  Enhanced Version
Camera + Voice Input | Voice Output | 40+ Items
Uses YOLOv8 + HSV Colour Analysis + Active Voice Commands
"""

import cv2
import numpy as np
import threading
import queue
import time
import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO

# ========== LOAD KNOWLEDGE BASE (40+ fruits & vegetables) ==========
# Keep the PRODUCE_INFO and VOICE_ALIASES dictionaries exactly as in your file.
# (They are already perfect.)  ↓↓↓
from produce_database import PRODUCE_INFO, VOICE_ALIASES, COCO_TO_KB

# Operating modes
MODES = ["camera", "voice", "both"]
current_mode = {"value": "both"}


# ========== TEXT-TO-SPEECH ENGINE ==========
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text, interrupt=True):
    if interrupt:
        with tts_queue.mutex:
            tts_queue.queue.clear()
    tts_queue.put(text)


# ========== DETECTION HELPERS ==========
def classify_by_colour(roi_bgr, threshold=1500):
    if roi_bgr is None or roi_bgr.size == 0:
        return []
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hits = []
    for name, info in PRODUCE_INFO.items():
        lo = np.array(info["skin"][0])
        hi = np.array(info["skin"][1])
        cnt = int(np.sum(cv2.inRange(hsv, lo, hi) > 0))
        if cnt >= threshold:
            hits.append((name, cnt))
    hits.sort(key=lambda x: -x[1])
    return hits

def estimate_ripeness(roi_bgr, name):
    if name not in PRODUCE_INFO or roi_bgr is None or roi_bgr.size == 0:
        return "Hard to tell"
    info = PRODUCE_INFO[name]
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    rp = int(np.sum(cv2.inRange(hsv, np.array(info["ripe_hsv"][0]), np.array(info["ripe_hsv"][1])) > 0))
    up = int(np.sum(cv2.inRange(hsv, np.array(info["unripe_hsv"][0]), np.array(info["unripe_hsv"][1])) > 0))
    if rp > up and rp > 300:
        return "Ripe"
    if up > 300:
        return "Unripe / Raw"
    return "Hard to tell"

def get_size_label(box_area, frame_area):
    r = box_area / max(frame_area, 1)
    if r > 0.3: return "Large"
    if r > 0.1: return "Medium"
    return "Small"


# ========== VOICE HANDLER ==========
voice_state = {"text":"", "active":False, "target":None, "mode_cmd":None, "stop_cmd":False}
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True
mic_available = True

def resolve_spoken(text):
    t = text.lower().strip()
    for alias, val in VOICE_ALIASES.items():
        if alias in t:
            if val.startswith("__mode_"):
                return None, val, False
            if val == "__stop__":
                return None, None, True
            return val, None, False
    for name in PRODUCE_INFO:
        if name in t:
            return name, None, False
    return None, None, False

def listen_loop():
    global mic_available
    try:
        mic = sr.Microphone()
    except OSError:
        mic_available = False
        return

    speak("System ready. Say any fruit or vegetable name or point it at the camera.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
            text = recognizer.recognize_google(audio).lower()
            voice_state["text"] = text
            voice_state["active"] = True
            produce, mode_cmd, stop = resolve_spoken(text)
            voice_state["target"], voice_state["mode_cmd"], voice_state["stop_cmd"] = produce, mode_cmd, stop
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            voice_state["text"] = "Could not understand"
        except sr.RequestError:
            voice_state["text"] = "Speech service unavailable"
        time.sleep(0.05)


# ========== DRAWING UTILITIES ==========
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_info(frame, name, ripeness, safe, info, conf, x1, y1, x2, y2, confirmed=False):
    color = (0,255,255) if confirmed else ((0,200,100) if safe else (0,0,255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = f"{name} ({ripeness}) {conf*100:.1f}%"
    cv2.putText(frame, txt, (x1, y1-8), FONT, 0.55, color, 1, cv2.LINE_AA)

def draw_hud(frame, mode, heard, target):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,h-40),(w,h),(0,0,0),-1)
    msg = f"Mode: {mode.upper()} | Heard: {heard[:40]} | Target: {target or '—'} | Q=quit M=mode"
    cv2.putText(frame,msg,(10,h-10),FONT,0.48,(0,255,0),1,cv2.LINE_AA)


# ========== YOLO MODEL ==========
model = YOLO("yolov8n.pt")

voice_thread = threading.Thread(target=listen_loop, daemon=True)
voice_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

last_spoken_key = ""
last_speak_time = 0

print("="*60)
print("Fruit & Vegetable Detector | Voice + Camera Operational")
print("Say 'camera mode', 'voice mode', or a produce name to control it.")
print("="*60)


# ========== MAIN LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_area = h*w
    mode = current_mode["value"]
    now = time.time()

    yolo_found = {}
    results = model(frame, conf=0.4, verbose=False)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label not in COCO_TO_KB:
                continue
            kb_name = COCO_TO_KB[label]
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            x1,y1 = max(x1,0), max(y1,0)
            roi = frame[y1:y2, x1:x2]
            ripeness = estimate_ripeness(roi, kb_name)
            yolo_found[kb_name]=(x1,y1,x2,y2,ripeness,conf)

    voice_target = voice_state["target"]
    if voice_state["active"]:
        voice_state["active"] = False
        if voice_state["stop_cmd"]:
            voice_state["target"]=None
            speak("Target cleared.")
        elif voice_state["mode_cmd"]:
            new_mode = voice_state["mode_cmd"].replace("__mode_","").replace("__","")
            current_mode["value"]=new_mode
            speak(f"Switched to {new_mode} mode.")
        elif voice_target:
            speak(f"Looking for {voice_target}. Please hold it in front of the camera.")

    colour_hits = classify_by_colour(frame, threshold=1500)
    colour_found = {n:c for n,c in colour_hits}

    for name,(x1,y1,x2,y2,ripeness,conf) in yolo_found.items():
        info = PRODUCE_INFO[name]
        confirmed = (voice_target == name)
        draw_info(frame,name,ripeness,info["safe"],info,conf,x1,y1,x2,y2,confirmed)

        key = f"{name}-{ripeness}"
        if now - last_speak_time > 6 and key != last_spoken_key:
            speak(f"{('Camera and voice confirmed' if confirmed else 'Detected')} {name}. It looks {ripeness.lower()}. {info['notes']}")
            last_spoken_key = key
            last_speak_time = now

    if voice_target and voice_target not in yolo_found:
        if voice_target in colour_found:
            ripeness = estimate_ripeness(frame, voice_target)
            if now - last_speak_time > 6:
                speak(f"Colour scan found {voice_target}. It seems {ripeness.lower()}. {PRODUCE_INFO[voice_target]['notes']}")
                last_spoken_key = f"{voice_target}-{ripeness}"
                last_speak_time = now

    draw_hud(frame, mode, voice_state["text"], voice_target)
    cv2.imshow("Fruit & Vegetable Detector | Voice + Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        idx = MODES.index(current_mode["value"])
        current_mode["value"]=MODES[(idx+1)%len(MODES)]
        speak(f"Switched to {current_mode['value']} mode.")

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)
