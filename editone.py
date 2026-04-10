import cv2
import numpy as np
import threading
import queue
import time

# pip install ultralytics opencv-python speechrecognition pyaudio pyttsx3
from ultralytics import YOLO
import speech_recognition as sr
import pyttsx3

# ═══════════════════════════════════════════════════════════════
#  MODE SYSTEM
#  MODE 1 — "camera"  : YOLO watches camera continuously, auto-detects
#  MODE 2 — "voice"   : User speaks a name, camera confirms via colour scan
#  MODE 3 — "both"    : Voice + YOLO camera detection run simultaneously
#  Default starts in MODE 3 (both)
# ═══════════════════════════════════════════════════════════════
MODES        = ["camera", "voice", "both"]
current_mode = {"value": "both"}   # shared across threads

# ═══════════════════════════════════════════════════════════════
#  TTS ENGINE  (own thread so it never blocks the camera loop)
# ═══════════════════════════════════════════════════════════════
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 155)
tts_queue  = queue.Queue()

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
    """Clear queue and speak immediately (non-blocking)."""
    with tts_queue.mutex:
        tts_queue.queue.clear()
    tts_queue.put(text)


# ═══════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE  (40+ fruits & vegetables)
# ═══════════════════════════════════════════════════════════════
PRODUCE_INFO = {
    # ── Fruits ──────────────────────────────────────────────────
    "apple":       {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([0,100,100],[15,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe raw/cooked. Avoid seeds — trace cyanide."},
    "banana":      {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([20,100,150],[35,255,255]),
                    "unripe_hsv":([35,60,150],[85,255,255]),
                    "notes":"Safe. Unripe may cause bloating."},
    "orange":      {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([10,150,150],[25,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe raw. High in vitamin C."},
    "mango":       {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([10,120,150],[30,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe. Unripe mango is sour but edible."},
    "grape":       {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([130,40,40],[160,255,200]),
                    "unripe_hsv":([35,60,80],[85,200,200]),
                    "notes":"Safe for humans. TOXIC to dogs."},
    "strawberry":  {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([0,150,100],[10,255,255]),
                    "unripe_hsv":([35,60,80],[85,255,255]),
                    "notes":"Safe raw. White/green ones are very sour."},
    "watermelon":  {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([0,150,100],[10,255,255]),
                    "unripe_hsv":([35,50,80],[85,200,200]),
                    "notes":"Safe raw. Rind edible when cooked."},
    "pineapple":   {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([20,100,150],[35,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe. Unripe can cause stomach upset."},
    "papaya":      {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([10,120,150],[25,255,255]),
                    "unripe_hsv":([35,60,100],[85,200,255]),
                    "notes":"Safe ripe. Unripe contains latex — avoid if pregnant."},
    "lemon":       {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([22,120,180],[35,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe. Very acidic — can erode tooth enamel."},
    "pomegranate": {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([0,120,80],[15,255,200]),
                    "unripe_hsv":([35,60,80],[85,200,200]),
                    "notes":"Safe. Eat seeds (arils) only, not the rind."},
    "kiwi":        {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([25,60,100],[40,180,200]),
                    "unripe_hsv":([35,80,100],[85,255,255]),
                    "notes":"Safe including skin. High in vitamin C and K."},
    "peach":       {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([8,100,180],[20,220,255]),
                    "unripe_hsv":([35,60,80],[85,200,200]),
                    "notes":"Safe. Pit contains amygdalin — discard it."},
    "pear":        {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([22,60,150],[40,200,255]),
                    "unripe_hsv":([35,80,80],[85,255,200]),
                    "notes":"Safe raw. High in fibre."},
    "cherry":      {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([0,150,60],[10,255,180]),
                    "unripe_hsv":([35,60,80],[85,200,200]),
                    "notes":"Safe. Pit has cyanogenic compounds — never chew it."},
    "coconut":     {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([15,30,60],[30,120,180]),
                    "unripe_hsv":([35,60,100],[85,200,255]),
                    "notes":"Safe. Young coconut water is very nutritious."},
    "guava":       {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([25,60,150],[45,200,255]),
                    "unripe_hsv":([35,80,80],[85,255,200]),
                    "notes":"Safe raw including skin and seeds."},
    "plum":        {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([130,60,60],[160,255,200]),
                    "unripe_hsv":([35,60,80],[85,200,200]),
                    "notes":"Safe. Pit is toxic — discard it."},
    "fig":         {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([130,40,60],[160,200,180]),
                    "unripe_hsv":([35,60,80],[85,200,200]),
                    "notes":"Safe. Milky sap in unripe figs irritates skin."},
    "avocado":     {"type":"Fruit",     "safe":True,
                    "ripe_hsv":  ([35,30,30],[85,150,120]),
                    "unripe_hsv":([35,80,100],[85,255,200]),
                    "notes":"Safe for humans. TOXIC to birds and most pets."},
    # ── Vegetables ──────────────────────────────────────────────
    "broccoli":    {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,80,50],[85,255,200]),
                    "unripe_hsv":([35,60,150],[85,200,255]),
                    "notes":"Safe raw/cooked. Cooking improves digestibility."},
    "carrot":      {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([8,150,150],[20,255,255]),
                    "unripe_hsv":([8,80,200],[20,180,255]),
                    "notes":"Safe raw/cooked. Rich in beta-carotene."},
    "tomato":      {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([0,150,100],[10,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe ripe. Unripe contains solanine — limit intake."},
    "potato":      {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([15,30,100],[30,120,200]),
                    "unripe_hsv":([35,40,80],[85,150,180]),
                    "notes":"Must be cooked. Green parts have solanine — AVOID."},
    "corn":        {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([20,100,150],[35,255,255]),
                    "unripe_hsv":([35,60,180],[85,180,255]),
                    "notes":"Safe raw or cooked."},
    "onion":       {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([15,40,150],[30,150,255]),
                    "unripe_hsv":([35,60,100],[85,200,255]),
                    "notes":"Safe for humans. TOXIC to dogs and cats."},
    "garlic":      {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([15,20,180],[30,80,255]),
                    "unripe_hsv":([35,30,150],[85,120,255]),
                    "notes":"Safe for humans. TOXIC to dogs and cats."},
    "spinach":     {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,80,50],[85,255,180]),
                    "unripe_hsv":([35,60,150],[85,200,255]),
                    "notes":"Safe raw/cooked. High in iron and oxalates."},
    "cucumber":    {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,60,80],[85,200,200]),
                    "unripe_hsv":([35,80,150],[85,255,255]),
                    "notes":"Safe raw or cooked. Very hydrating."},
    "eggplant":    {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([125,60,40],[155,255,180]),
                    "unripe_hsv":([35,40,80],[125,150,180]),
                    "notes":"Cook before eating. Raw contains solanine."},
    "capsicum":    {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([0,150,100],[15,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe raw/cooked. High in vitamin C."},
    "cabbage":     {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,40,80],[85,180,200]),
                    "unripe_hsv":([35,60,150],[85,255,255]),
                    "notes":"Safe raw/cooked. Can cause gas in large amounts."},
    "cauliflower": {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([15,10,200],[35,60,255]),
                    "unripe_hsv":([35,30,180],[85,120,255]),
                    "notes":"Safe raw or cooked."},
    "peas":        {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,80,80],[85,200,200]),
                    "unripe_hsv":([35,60,150],[85,255,255]),
                    "notes":"Safe raw or cooked. High in protein."},
    "mushroom":    {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([15,20,60],[30,80,180]),
                    "unripe_hsv":([15,10,150],[30,50,255]),
                    "notes":"Only eat known edible varieties. Wild ones can be DEADLY."},
    "ginger":      {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([15,40,150],[30,120,230]),
                    "unripe_hsv":([35,40,150],[85,120,255]),
                    "notes":"Safe raw/cooked. Anti-inflammatory properties."},
    "radish":      {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([0,120,80],[15,255,220]),
                    "unripe_hsv":([35,60,80],[85,200,200]),
                    "notes":"Safe raw or cooked. Peppery flavour when raw."},
    "beetroot":    {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([160,80,40],[180,255,180]),
                    "unripe_hsv":([35,40,80],[85,150,180]),
                    "notes":"Safe raw/cooked. May turn urine red — harmless."},
    "sweet potato":{"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([8,100,150],[20,220,255]),
                    "unripe_hsv":([8,60,180],[20,150,255]),
                    "notes":"Best cooked. Raw is hard to digest."},
    "bitter gourd":{"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,80,80],[85,200,200]),
                    "unripe_hsv":([35,80,130],[85,255,255]),
                    "notes":"Safe. Red seeds cause vomiting — discard them."},
    "bottle gourd":{"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,60,80],[85,180,200]),
                    "unripe_hsv":([35,80,120],[85,255,255]),
                    "notes":"Cook before eating. Bitter taste = toxic — DISCARD."},
    "chili":       {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([0,150,100],[15,255,255]),
                    "unripe_hsv":([35,80,80],[85,255,255]),
                    "notes":"Safe. Capsaicin irritates eyes/skin — use gloves."},
    "drumstick":   {"type":"Vegetable", "safe":True,
                    "ripe_hsv":  ([35,40,80],[85,150,200]),
                    "unripe_hsv":([35,60,120],[85,200,255]),
                    "notes":"Safe cooked. Leaves, pods, seeds all edible."},
}

VOICE_ALIASES = {
    "bell pepper":"capsicum","pepper":"capsicum",
    "hot pepper":"chili","red chili":"chili","green chili":"chili",
    "yam":"sweet potato","beet":"beetroot","beets":"beetroot",
    "zucchini":"cucumber","courgette":"cucumber",
    "brinjal":"eggplant","aubergine":"eggplant",
    "lady finger":"drumstick","okra":"drumstick",
    "grapes":"grape","cherries":"cherry","strawberries":"strawberry",
    "mangoes":"mango","tomatoes":"tomato","potatoes":"potato",
    "mushrooms":"mushroom","onions":"onion","carrots":"carrot","lemons":"lemon",
    # mode-switching phrases
    "camera mode":"__mode_camera__",
    "voice mode":"__mode_voice__",
    "both mode":"__mode_both__",
    "use both":"__mode_both__",
    "switch to camera":"__mode_camera__",
    "switch to voice":"__mode_voice__",
    "show all":"__mode_both__",
}

COCO_TO_KB = {
    "apple":"apple","banana":"banana","orange":"orange",
    "broccoli":"broccoli","carrot":"carrot",
}


# ═══════════════════════════════════════════════════════════════
#  VOICE LISTENER  (background thread)
# ═══════════════════════════════════════════════════════════════
voice_state = {
    "text":   "",
    "active": False,
    "target": None,
    "mode_cmd": None,      # filled when user says a mode-switch phrase
}
recognizer = sr.Recognizer()
recognizer.energy_threshold         = 300
recognizer.dynamic_energy_threshold = True
mic_available = True

def resolve_spoken(text):
    """Return (produce_name_or_None, mode_command_or_None)."""
    text = text.lower().strip()
    # check aliases first (includes mode commands)
    for alias, val in VOICE_ALIASES.items():
        if alias in text:
            if val.startswith("__mode_"):
                return None, val
            return val, None
    # direct produce names
    for name in PRODUCE_INFO:
        if name in text:
            return name, None
    return None, None

def listen_loop():
    global mic_available
    try:
        mic = sr.Microphone()
    except OSError:
        mic_available = False
        voice_state["text"] = "No microphone found"
        return

    speak("Detector ready. Show produce to camera, say its name, or use both together.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
            text = recognizer.recognize_google(audio).lower()
            voice_state["text"]   = text
            voice_state["active"] = True

            produce, mode_cmd = resolve_spoken(text)
            voice_state["target"]   = produce
            voice_state["mode_cmd"] = mode_cmd

            if mode_cmd:
                new_mode = mode_cmd.replace("__mode_","").replace("__","")
                current_mode["value"] = new_mode
                speak(f"Switched to {new_mode} mode.")
            elif produce:
                speak(f"Looking for {produce} on camera.")
            else:
                speak("I didn't catch a produce name. Try again.")

        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            voice_state["text"] = "Could not understand audio"
        except sr.RequestError:
            voice_state["text"] = "Speech service unavailable"
        except Exception as e:
            voice_state["text"] = str(e)
        time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def estimate_ripeness(roi_bgr, name):
    if name not in PRODUCE_INFO or roi_bgr.size == 0:
        return "Unknown"
    info = PRODUCE_INFO[name]
    hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    r_lo, r_hi = np.array(info["ripe_hsv"][0]),   np.array(info["ripe_hsv"][1])
    u_lo, u_hi = np.array(info["unripe_hsv"][0]), np.array(info["unripe_hsv"][1])
    rp = np.sum(cv2.inRange(hsv, r_lo, r_hi) > 0)
    up = np.sum(cv2.inRange(hsv, u_lo, u_hi) > 0)
    if rp > up and rp > 300:   return "Ripe"
    if up > 300:               return "Unripe / Raw"
    return "Hard to tell"

def get_size_label(box_area, frame_area):
    r = box_area / frame_area
    if r > 0.30: return "Large"
    if r > 0.10: return "Medium"
    return "Small"

def draw_panel(frame, x1, y1, x2, y2, name, ripeness, size, info, conf,
               highlighted=False, voice_confirmed=False):
    """Draw bounding box + info panel over a detection."""
    if voice_confirmed:
        box_color = (0, 255, 255)   # bright cyan — voice + camera match
        thickness = 3
    elif highlighted:
        box_color = (0, 200, 255)   # softer cyan — voice targeted
        thickness = 3
    elif info["safe"]:
        box_color = (0, 220, 100)   # green — safe
        thickness = 2
    else:
        box_color = (0, 60, 220)    # red — caution
        thickness = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
    if voice_confirmed:
        # extra glowing border
        cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (255, 255, 0), 1)

    ph  = 145
    px  = max(x1, 0)
    py  = max(y1 - ph, 0)
    ov  = frame.copy()
    cv2.rectangle(ov, (px, py), (px + 310, py + ph), (12, 12, 12), -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    tx, ty = px + 8, py + 20

    safe_str   = "SAFE TO EAT" if info["safe"] else "CAUTION"
    safe_color = (80, 255, 130) if info["safe"] else (80, 110, 255)
    src_label  = "[VOICE+CAM]" if voice_confirmed else ("[VOICE]" if highlighted else "[CAM]")

    lines = [
        (f"{name.upper()}  {src_label}",          (255, 255, 255),  0.52, 1),
        (f"Type     : {info['type']}",             (200, 200, 200),  0.44, 1),
        (f"Ripeness : {ripeness}",                 (100, 220, 255),  0.44, 1),
        (f"Size     : {size}",                     (200, 200, 200),  0.44, 1),
        (f"Safety   : {safe_str}",                 safe_color,       0.44, 1),
        (f"{info['notes'][:45]}",                  (170, 170, 170),  0.37, 1),
    ]
    for text, color, scale, thick in lines:
        cv2.putText(frame, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
        ty += 23

    cv2.putText(frame, f"{conf:.0%}", (x2 - 50, y2 - 6),
                font, 0.48, (255, 255, 160), 1, cv2.LINE_AA)


def voice_only_scan(frame, target_name):
    """
    VOICE MODE or fallback when YOLO misses the spoken produce.
    Scans the entire frame with colour analysis.
    Returns ripeness string or None if not found.
    """
    if target_name not in PRODUCE_INFO:
        return None
    info = PRODUCE_INFO[target_name]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r_lo, r_hi = np.array(info["ripe_hsv"][0]),   np.array(info["ripe_hsv"][1])
    u_lo, u_hi = np.array(info["unripe_hsv"][0]), np.array(info["unripe_hsv"][1])
    rp = np.sum(cv2.inRange(hsv, r_lo, r_hi) > 0)
    up = np.sum(cv2.inRange(hsv, u_lo, u_hi) > 0)

    if rp > 2000:   ripeness = "Ripe"
    elif up > 2000: ripeness = "Unripe / Raw"
    else:           return None

    return ripeness


def draw_voice_overlay(frame, target_name, ripeness):
    """Draw full-frame overlay when running in VOICE or fallback mode."""
    info = PRODUCE_INFO.get(target_name, {})
    h, w = frame.shape[:2]

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 180, 255), -1)
    cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)
    cv2.rectangle(frame, (16, 16), (w - 16, h - 16), (0, 200, 255), 2)

    safe_str   = "SAFE TO EAT" if info.get("safe") else "CAUTION"
    safe_color = (0, 255, 130) if info.get("safe") else (0, 80, 255)

    lines = [
        (f"VOICE TARGET : {target_name.upper()}",              (0, 220, 255),  0.65, 2),
        (f"Type         : {info.get('type','?')}",             (220, 220, 220),0.52, 1),
        (f"Ripeness     : {ripeness}",                         (100, 220, 255),0.52, 1),
        (f"Safety       : {safe_str}",                         safe_color,     0.52, 1),
        (f"Note         : {info.get('notes','')[:52]}",        (180, 180, 180),0.44, 1),
    ]
    y = 70
    for text, color, scale, thick in lines:
        cv2.putText(frame, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += 32


def draw_hud(frame, mode, heard, target):
    """Bottom status bar + mode indicator."""
    h, w = frame.shape[:2]

    # bottom bar
    cv2.rectangle(frame, (0, h - 44), (w, h), (15, 15, 15), -1)
    mic_icon = "MIC ON" if mic_available else "NO MIC"
    heard_t  = heard[:50] if heard else "Listening..."
    target_t = target or "—"
    cv2.putText(frame,
                f"[{mic_icon}]  Heard: {heard_t}   |   Target: {target_t}   |   Q=quit  M=mode",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 160), 1, cv2.LINE_AA)

    # mode badge top-right
    mode_colors = {"camera": (80, 200, 80), "voice": (80, 180, 255), "both": (0, 220, 255)}
    mc = mode_colors.get(mode, (200, 200, 200))
    badge = f"MODE: {mode.upper()}"
    bw, _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0], None
    cv2.rectangle(frame, (w - bw[0] - 20, 8), (w - 8, 34), (20, 20, 20), -1)
    cv2.putText(frame, badge, (w - bw[0] - 14, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, mc, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════
model = YOLO("yolov8n.pt")

voice_thread = threading.Thread(target=listen_loop, daemon=True)
voice_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_spoken_key = ""
speak_cooldown  = 0

print("=" * 60)
print("  Fruit & Vegetable Detector  |  DUAL INPUT (Voice + Camera)")
print("  Q = quit     M = cycle mode (camera / voice / both)")
print("  Voice commands: say produce name, or say:")
print("    'camera mode' / 'voice mode' / 'both mode'")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    frame_area = frame.shape[0] * frame.shape[1]
    mode       = current_mode["value"]

    yolo_found   = {}   # kb_name → (x1,y1,x2,y2, ripeness, size, conf)
    voice_target = voice_state["target"]

    # ── CAMERA DETECTION (YOLO) ─────────────────────────────────
    if mode in ("camera", "both"):
        results = model(frame, conf=0.45, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                if label not in COCO_TO_KB:
                    continue
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi    = frame[max(0,y1):y2, max(0,x1):x2]
                kb_name  = COCO_TO_KB[label]
                ripeness = estimate_ripeness(roi, kb_name)
                size     = get_size_label((x2-x1)*(y2-y1), frame_area)
                yolo_found[kb_name] = (x1, y1, x2, y2, ripeness, size, conf)

        # draw each YOLO detection
        for kb_name, vals in yolo_found.items():
            x1, y1, x2, y2, ripeness, size, conf = vals
            info      = PRODUCE_INFO[kb_name]
            highlight = (voice_target == kb_name)
            confirmed = highlight   # voice + camera agree

            draw_panel(frame, x1, y1, x2, y2,
                       kb_name, ripeness, size, info, conf,
                       highlighted=highlight,
                       voice_confirmed=confirmed)

            # auto-speak with cooldown
            key = f"{kb_name}-{ripeness}"
            if key != last_spoken_key and time.time() > speak_cooldown:
                prefix = "Voice confirmed!" if confirmed else "Detected"
                speak(f"{prefix} {kb_name}. {ripeness}. {info['notes']}")
                last_spoken_key = key
                speak_cooldown  = time.time() + 6

    # ── VOICE INPUT PROCESSING ──────────────────────────────────
    if voice_state["active"] and mode in ("voice", "both"):
        voice_state["active"] = False   # consume the event

        if voice_target:
            if voice_target not in yolo_found:
                # YOLO missed it → fall back to colour scan
                ripeness = voice_only_scan(frame, voice_target)
                if ripeness:
                    draw_voice_overlay(frame, voice_target, ripeness)
                    info = PRODUCE_INFO[voice_target]
                    key  = f"voice-{voice_target}-{ripeness}"
                    if key != last_spoken_key:
                        speak(f"Colour scan: {voice_target} looks {ripeness}. {info['notes']}")
                        last_spoken_key = key
                        speak_cooldown  = time.time() + 6
                else:
                    speak(f"Cannot clearly see {voice_target} in the frame. Hold it closer.")

    elif voice_state["active"]:
        voice_state["active"] = False   # clear even if mode doesn't use it

    # ── VOICE-ONLY MODE: always run scan for current target ─────
    if mode == "voice" and voice_target:
        ripeness = voice_only_scan(frame, voice_target)
        if ripeness:
            draw_voice_overlay(frame, voice_target, ripeness)

    # ── HUD ────────────────────────────────────────────────────
    draw_hud(frame, mode, voice_state["text"], voice_target)

    cv2.imshow("Fruit & Vegetable Detector  |  Voice + Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        # cycle mode manually with M key
        idx = MODES.index(current_mode["value"])
        current_mode["value"] = MODES[(idx + 1) % len(MODES)]
        speak(f"Mode: {current_mode['value']}")

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)