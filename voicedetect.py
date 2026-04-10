import cv2
import numpy as np
import threading
import queue
import time

# pip install ultralytics opencv-python speechrecognition pyaudio pyttsx3
from ultralytics import YOLO
import speech_recognition as sr
import pyttsx3

# ─────────────────────────────────────────────
#  TTS engine (runs in its own thread to avoid blocking)
# ─────────────────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)
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

def speak(text):
    """Non-blocking speak."""
    with tts_queue.mutex:
        tts_queue.queue.clear()   # cancel any pending speech
    tts_queue.put(text)


# ─────────────────────────────────────────────
#  Knowledge base
# ─────────────────────────────────────────────
PRODUCE_INFO = {
    # ── Fruits ──────────────────────────────
    "apple": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([0,   100, 100], [15,  255, 255]),
        "unripe_hsv": ([35,  80,  80],  [85,  255, 255]),
        "notes": "Safe raw or cooked. Avoid seeds — they contain trace cyanide.",
    },
    "banana": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([20, 100, 150], [35, 255, 255]),
        "unripe_hsv": ([35,  60, 150], [85, 255, 255]),
        "notes": "Safe. Unripe bananas may cause bloating.",
    },
    "orange": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([10, 150, 150], [25, 255, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 255]),
        "notes": "Safe raw. High in vitamin C.",
    },
    "mango": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([10, 120, 150], [30, 255, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 255]),
        "notes": "Safe. Unripe mango is sour but edible.",
    },
    "grape": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([130, 40, 40], [160, 255, 200]),
        "unripe_hsv": ([35,  60, 80], [85,  200, 200]),
        "notes": "Safe for humans. TOXIC to dogs.",
    },
    "strawberry": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([0,  150, 100], [10, 255, 255]),
        "unripe_hsv": ([35,  60,  80], [85, 255, 255]),
        "notes": "Safe raw. White/green ones are unripe and very sour.",
    },
    "watermelon": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([0,  150, 100], [10, 255, 255]),   # red flesh
        "unripe_hsv": ([35,  50,  80], [85, 200, 200]),
        "notes": "Safe raw. Rind is also edible when cooked.",
    },
    "pineapple": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([20, 100, 150], [35, 255, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 255]),
        "notes": "Safe. Unripe pineapple can cause stomach upset.",
    },
    "papaya": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([10, 120, 150], [25, 255, 255]),
        "unripe_hsv": ([35,  60, 100], [85, 200, 255]),
        "notes": "Safe when ripe. Unripe papaya contains latex — avoid if pregnant.",
    },
    "lemon": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([22, 120, 180], [35, 255, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 255]),
        "notes": "Safe. Very acidic — can erode tooth enamel.",
    },
    "pomegranate": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([0,  120,  80], [15, 255, 200]),
        "unripe_hsv": ([35,  60,  80], [85, 200, 200]),
        "notes": "Safe. Only eat the seeds (arils), not the rind.",
    },
    "kiwi": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([25,  60, 100], [40, 180, 200]),
        "unripe_hsv": ([35,  80, 100], [85, 255, 255]),
        "notes": "Safe raw including skin. High in vitamin C and K.",
    },
    "peach": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([8,  100, 180], [20, 220, 255]),
        "unripe_hsv": ([35,  60,  80], [85, 200, 200]),
        "notes": "Safe. Pit contains amygdalin — do not eat it.",
    },
    "pear": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([22,  60, 150], [40, 200, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 200]),
        "notes": "Safe raw. High in fibre.",
    },
    "cherry": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([0,  150,  60], [10, 255, 180]),
        "unripe_hsv": ([35,  60,  80], [85, 200, 200]),
        "notes": "Safe. Pit contains cyanogenic compounds — do not chew it.",
    },
    "coconut": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([15,  30,  60], [30, 120, 180]),
        "unripe_hsv": ([35,  60, 100], [85, 200, 255]),
        "notes": "Safe. Young coconut water is nutritious.",
    },
    "guava": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([25,  60, 150], [45, 200, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 200]),
        "notes": "Safe raw including skin and seeds.",
    },
    "plum": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([130, 60,  60], [160, 255, 200]),
        "unripe_hsv": ([35,  60,  80], [85, 200, 200]),
        "notes": "Safe. Pit is toxic — discard it.",
    },
    "fig": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([130, 40,  60], [160, 200, 180]),
        "unripe_hsv": ([35,  60,  80], [85, 200, 200]),
        "notes": "Safe. Milky sap in unripe figs can irritate skin.",
    },
    "avocado": {
        "type": "Fruit", "safe": True,
        "ripe_hsv":   ([35,  30,  30], [85, 150, 120]),  # dark green/black
        "unripe_hsv": ([35,  80, 100], [85, 255, 200]),
        "notes": "Safe for humans. TOXIC to birds and most pets.",
    },

    # ── Vegetables ──────────────────────────
    "broccoli": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  80,  50], [85, 255, 200]),
        "unripe_hsv": ([35,  60, 150], [85, 200, 255]),
        "notes": "Safe raw or cooked. Cooking improves digestibility.",
    },
    "carrot": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([8,  150, 150], [20, 255, 255]),
        "unripe_hsv": ([8,   80, 200], [20, 180, 255]),
        "notes": "Safe raw or cooked. Rich in beta-carotene.",
    },
    "tomato": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([0,  150, 100], [10, 255, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 255]),
        "notes": "Safe when ripe. Unripe/green tomatoes contain solanine — limit intake.",
    },
    "potato": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([15,  30, 100], [30, 120, 200]),
        "unripe_hsv": ([35,  40,  80], [85, 150, 180]),
        "notes": "Must be cooked. Green parts contain solanine — AVOID.",
    },
    "corn": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([20, 100, 150], [35, 255, 255]),
        "unripe_hsv": ([35,  60, 180], [85, 180, 255]),
        "notes": "Safe raw or cooked.",
    },
    "onion": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([15,  40, 150], [30, 150, 255]),
        "unripe_hsv": ([35,  60, 100], [85, 200, 255]),
        "notes": "Safe for humans. TOXIC to dogs and cats.",
    },
    "garlic": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([15,  20, 180], [30,  80, 255]),
        "unripe_hsv": ([35,  30, 150], [85, 120, 255]),
        "notes": "Safe for humans. TOXIC to dogs and cats.",
    },
    "spinach": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  80,  50], [85, 255, 180]),
        "unripe_hsv": ([35,  60, 150], [85, 200, 255]),
        "notes": "Safe raw or cooked. High in iron and oxalates.",
    },
    "cucumber": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  60,  80], [85, 200, 200]),
        "unripe_hsv": ([35,  80, 150], [85, 255, 255]),
        "notes": "Safe raw or cooked. Very hydrating.",
    },
    "eggplant": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([125, 60,  40], [155, 255, 180]),
        "unripe_hsv": ([35,  40,  80], [125, 150, 180]),
        "notes": "Cook before eating. Raw eggplant contains solanine.",
    },
    "capsicum": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([0,  150, 100], [15, 255, 255]),   # red when ripe
        "unripe_hsv": ([35,  80,  80], [85, 255, 255]),   # green when unripe
        "notes": "Safe raw or cooked. High in vitamin C.",
    },
    "cabbage": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  40,  80], [85, 180, 200]),
        "unripe_hsv": ([35,  60, 150], [85, 255, 255]),
        "notes": "Safe raw or cooked. Can cause gas in large amounts.",
    },
    "cauliflower": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([15,  10, 200], [35,  60, 255]),  # off-white
        "unripe_hsv": ([35,  30, 180], [85, 120, 255]),
        "notes": "Safe raw or cooked.",
    },
    "peas": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  80,  80], [85, 200, 200]),
        "unripe_hsv": ([35,  60, 150], [85, 255, 255]),
        "notes": "Safe raw or cooked. High in protein.",
    },
    "mushroom": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([15,  20,  60], [30,  80, 180]),
        "unripe_hsv": ([15,  10, 150], [30,  50, 255]),
        "notes": "Only eat known edible varieties. Wild mushrooms can be DEADLY.",
    },
    "ginger": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([15,  40, 150], [30, 120, 230]),
        "unripe_hsv": ([35,  40, 150], [85, 120, 255]),
        "notes": "Safe raw or cooked. Has anti-inflammatory properties.",
    },
    "radish": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([0,  120,  80], [15, 255, 220]),
        "unripe_hsv": ([35,  60,  80], [85, 200, 200]),
        "notes": "Safe raw or cooked. Peppery flavour when raw.",
    },
    "beetroot": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([160, 80,  40], [180, 255, 180]),
        "unripe_hsv": ([35,  40,  80], [85, 150, 180]),
        "notes": "Safe raw or cooked. May turn urine/stool red — harmless.",
    },
    "sweet potato": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([8,  100, 150], [20, 220, 255]),
        "unripe_hsv": ([8,   60, 180], [20, 150, 255]),
        "notes": "Best cooked. Raw is hard to digest.",
    },
    "bitter gourd": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  80,  80], [85, 200, 200]),
        "unripe_hsv": ([35,  80, 130], [85, 255, 255]),
        "notes": "Safe. Do not eat the red seeds — they can cause vomiting.",
    },
    "bottle gourd": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  60,  80], [85, 180, 200]),
        "unripe_hsv": ([35,  80, 120], [85, 255, 255]),
        "notes": "Cook before eating. Bitter taste indicates toxicity — DISCARD.",
    },
    "chili": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([0,  150, 100], [15, 255, 255]),
        "unripe_hsv": ([35,  80,  80], [85, 255, 255]),
        "notes": "Safe. Contains capsaicin — irritates eyes/skin. Use gloves.",
    },
    "drumstick": {
        "type": "Vegetable", "safe": True,
        "ripe_hsv":   ([35,  40,  80], [85, 150, 200]),
        "unripe_hsv": ([35,  60, 120], [85, 200, 255]),
        "notes": "Safe cooked. Leaves, pods, and seeds are all edible.",
    },
}

# Aliases for voice recognition (handles slight name variations)
VOICE_ALIASES = {
    "bell pepper": "capsicum", "pepper": "capsicum",
    "hot pepper": "chili", "red chili": "chili", "green chili": "chili",
    "yam": "sweet potato", "beet": "beetroot", "beets": "beetroot",
    "zucchini": "cucumber", "courgette": "cucumber",
    "eggplant": "eggplant", "brinjal": "eggplant", "aubergine": "eggplant",
    "lady finger": "drumstick", "okra": "drumstick",
    "grapes": "grape", "cherries": "cherry",
    "strawberries": "strawberry", "mangoes": "mango",
    "tomatoes": "tomato", "potatoes": "potato",
    "mushrooms": "mushroom", "onions": "onion",
    "carrots": "carrot", "lemons": "lemon",
}

# YOLO COCO → KB mapping
COCO_TO_KB = {
    "apple": "apple", "banana": "banana", "orange": "orange",
    "broccoli": "broccoli", "carrot": "carrot",
}


# ─────────────────────────────────────────────
#  Voice listener — runs in a background thread
# ─────────────────────────────────────────────
voice_command  = {"text": "", "active": False, "target": None}
recognizer     = sr.Recognizer()
recognizer.energy_threshold        = 300
recognizer.dynamic_energy_threshold = True
mic_available  = True

def resolve_produce_name(text):
    """Find the best matching produce name in spoken text."""
    text = text.lower().strip()
    # check aliases first
    for alias, real in VOICE_ALIASES.items():
        if alias in text:
            return real
    # check direct names
    for name in PRODUCE_INFO:
        if name in text:
            return name
    return None

def listen_loop():
    global mic_available
    try:
        mic = sr.Microphone()
    except OSError:
        mic_available = False
        voice_command["text"] = "No microphone found"
        return

    speak("Fruit and vegetable detector ready. Say a produce name to analyse it.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio).lower()
            voice_command["text"]   = text
            voice_command["active"] = True

            target = resolve_produce_name(text)
            voice_command["target"] = target

            if target:
                speak(f"Looking for {target} on camera.")
            else:
                speak("Sorry, I didn't catch a produce name. Try again.")
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            voice_command["text"] = "Could not understand audio"
        except sr.RequestError:
            voice_command["text"] = "Speech service unavailable"
        except Exception as e:
            voice_command["text"] = str(e)
        time.sleep(0.1)


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────
def estimate_ripeness(roi_bgr, name):
    if name not in PRODUCE_INFO or roi_bgr.size == 0:
        return "Unknown"
    info  = PRODUCE_INFO[name]
    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    r_lo, r_hi = np.array(info["ripe_hsv"][0]),   np.array(info["ripe_hsv"][1])
    u_lo, u_hi = np.array(info["unripe_hsv"][0]), np.array(info["unripe_hsv"][1])

    ripe_px   = np.sum(cv2.inRange(hsv, r_lo, r_hi) > 0)
    unripe_px = np.sum(cv2.inRange(hsv, u_lo, u_hi) > 0)

    if ripe_px > unripe_px and ripe_px > 300:
        return "Ripe"
    elif unripe_px > 300:
        return "Unripe / Raw"
    return "Hard to tell"

def get_size_label(box_area, frame_area):
    r = box_area / frame_area
    if r > 0.30:  return "Large"
    if r > 0.10:  return "Medium"
    return "Small"

def draw_panel(frame, x1, y1, x2, y2, name, ripeness, size, info, conf, highlighted=False):
    box_color = (0, 220, 100) if info["safe"] else (0, 60, 220)
    if highlighted:
        box_color = (0, 200, 255)    # cyan highlight for voice-targeted item
        cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), box_color, 3)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    ph   = 130
    px   = x1
    py   = max(y1 - ph, 0)
    ov   = frame.copy()
    cv2.rectangle(ov, (px, py), (px + 300, py + ph), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    tx, ty = px + 7, py + 20

    safe_str   = "SAFE TO EAT" if info["safe"] else "⚠ CAUTION"
    safe_color = (80, 255, 130) if info["safe"] else (80, 110, 255)

    lines = [
        (f"{name.upper()}  [{info['type']}]",  (255, 255, 255), 0.52, 1),
        (f"Ripeness : {ripeness}",              (100, 220, 255), 0.44, 1),
        (f"Size     : {size}",                  (200, 200, 200), 0.44, 1),
        (f"Safety   : {safe_str}",              safe_color,      0.44, 1),
        (f"{info['notes'][:42]}",               (170, 170, 170), 0.37, 1),
    ]
    for text, color, scale, thick in lines:
        cv2.putText(frame, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
        ty += 24

    cv2.putText(frame, f"{conf:.0%}", (x2 - 48, y2 - 6),
                font, 0.48, (255, 255, 160), 1, cv2.LINE_AA)


def voice_scan_roi(frame, target_name):
    """
    When user names a produce via voice and YOLO didn't find it,
    do a whole-frame colour scan and report what we find.
    """
    if target_name not in PRODUCE_INFO:
        return
    info = PRODUCE_INFO[target_name]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    r_lo, r_hi = np.array(info["ripe_hsv"][0]),   np.array(info["ripe_hsv"][1])
    u_lo, u_hi = np.array(info["unripe_hsv"][0]), np.array(info["unripe_hsv"][1])

    ripe_px   = np.sum(cv2.inRange(hsv, r_lo, r_hi) > 0)
    unripe_px = np.sum(cv2.inRange(hsv, u_lo, u_hi) > 0)
    threshold = 2000

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    if ripe_px > threshold:
        ripeness = "Ripe"
        msg = f"I can see what looks like a ripe {target_name} based on its color."
    elif unripe_px > threshold:
        ripeness = "Unripe / Raw"
        msg = f"I can see what looks like an unripe {target_name} based on its color."
    else:
        msg = f"I cannot clearly see a {target_name} in the frame right now."
        speak(msg)
        return

    # Draw a full-frame soft highlight
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (w - 20, h - 20), (0, 200, 255), 4)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    safe_str = "SAFE TO EAT" if info["safe"] else "CAUTION"
    panel_lines = [
        f"Voice target : {target_name.upper()}",
        f"Type         : {info['type']}",
        f"Ripeness     : {ripeness}",
        f"Safety       : {safe_str}",
        f"Note         : {info['notes'][:50]}",
    ]
    y_off = 80
    for line in panel_lines:
        cv2.putText(frame, line, (30, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 220, 255), 1, cv2.LINE_AA)
        y_off += 26

    speak(msg + f" It appears {ripeness.lower()}. {info['notes']}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# Start voice listener thread
voice_thread = threading.Thread(target=listen_loop, daemon=True)
voice_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_spoken_detection = ""
speak_cooldown = 0

print("=" * 55)
print("  Fruit & Vegetable Detector  |  Voice + Camera")
print("  Say a produce name to focus on it.")
print("  Press  Q  to quit.")
print("=" * 55)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    frame_area = frame.shape[0] * frame.shape[1]

    # ── Run YOLO ──────────────────────────────
    results         = model(frame, conf=0.45, verbose=False)
    yolo_found_names = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            if label not in COCO_TO_KB:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi  = frame[max(0,y1):y2, max(0,x1):x2]

            kb_name  = COCO_TO_KB[label]
            info     = PRODUCE_INFO[kb_name]
            ripeness = estimate_ripeness(roi, kb_name)
            size     = get_size_label((x2-x1)*(y2-y1), frame_area)

            # Highlight if it matches the voice target
            highlighted = (voice_command["target"] == kb_name)
            draw_panel(frame, x1, y1, x2, y2,
                       kb_name, ripeness, size, info, conf, highlighted)

            yolo_found_names.append(kb_name)

            # Auto-speak new detections (with cooldown)
            detection_key = f"{kb_name}-{ripeness}"
            if detection_key != last_spoken_detection and time.time() > speak_cooldown:
                speak(f"Detected {kb_name}. It is {ripeness}. {info['notes']}")
                last_spoken_detection = detection_key
                speak_cooldown = time.time() + 6   # 6 s cooldown

    # ── Voice-command overlay ─────────────────
    if voice_command["active"]:
        target = voice_command["target"]

        # If YOLO didn't find the named produce, fall back to colour scan
        if target and target not in yolo_found_names:
            voice_scan_roi(frame, target)

        # Reset flag so we don't repeat on every frame
        voice_command["active"] = False

    # ── Status bar ───────────────────────────
    h, w = frame.shape[:2]
    bar_y = h - 40
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)

    mic_icon = "MIC ON" if mic_available else "NO MIC"
    heard    = voice_command["text"][:55] if voice_command["text"] else "Listening..."
    target   = voice_command["target"] or "—"

    cv2.putText(frame, f"[{mic_icon}]  Heard: {heard}   |   Target: {target}   |   Q = quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 220, 160), 1, cv2.LINE_AA)

    cv2.imshow("Fruit & Vegetable Detector — Voice + Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)   # stop TTS thread cleanly