"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        FRUIT & VEGETABLE DETECTOR  —  FULL PRODUCE EDITION                 ║
║        Camera + Voice Input  |  Voice Output  |  40+ Items                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  HOW IT WORKS                                                               ║
║  ─────────────                                                              ║
║  LAYER 1 — YOLOv8  : Detects objects YOLO was trained on (apple, banana,   ║
║             orange, broccoli, carrot, etc.)                                 ║
║  LAYER 2 — Colour-Signature Classifier : Scans the whole frame (or a       ║
║             voice-targeted region) for HSV colour fingerprints of ALL 40+  ║
║             fruits/vegetables.  Kicks in whenever YOLO misses an item.     ║
║  LAYER 3 — Voice : Say any produce name.  System actively looks for it     ║
║             on camera and speaks back what it found.                        ║
║                                                                             ║
║  CONTROLS                                                                   ║
║  Q = quit   M = cycle mode (camera / voice / both)                         ║
║  Voice: "camera mode" / "voice mode" / "both mode" / "stop" / "reset"      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Install deps (once):
    pip install ultralytics opencv-python speechrecognition pyaudio pyttsx3
    # macOS  → pip install pyobjc  (for pyttsx3)
    # Linux  → sudo apt-get install espeak  (for pyttsx3)
"""

import cv2
import numpy as np
import threading
import queue
import time

from ultralytics import YOLO
import speech_recognition as sr
import pyttsx3


# ═══════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE  — 40+ fruits & vegetables
#  Each entry has:
#   type      : Fruit | Vegetable
#   safe      : bool — safe to eat?
#   ripe_hsv  : ([H_lo,S_lo,V_lo], [H_hi,S_hi,V_hi])
#   unripe_hsv: same
#   notes     : spoken safety tip
#   skin      : dominant external skin HSV range (used by frame-level scan)
# ═══════════════════════════════════════════════════════════════
PRODUCE_INFO = {
    # ── FRUITS ──────────────────────────────────────────────────────────────
    "apple":        {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([0,100,100],  [15,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([0,80,80],    [15,255,255]),
        "notes":"Safe raw or cooked. Avoid seeds — contain trace cyanide.",
    },
    "banana":       {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([20,100,150], [35,255,255]),
        "unripe_hsv": ([35,60,150],  [85,255,255]),
        "skin":       ([20,80,180],  [35,255,255]),
        "notes":"Safe. Unripe banana may cause bloating.",
    },
    "orange":       {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([10,150,150], [25,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([10,140,160], [22,255,255]),
        "notes":"Safe raw. High in vitamin C.",
    },
    "mango":        {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([10,120,150], [30,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([12,100,160], [28,255,255]),
        "notes":"Safe. Unripe mango is sour but edible.",
    },
    "grape":        {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([130,40,40],  [160,255,200]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([125,30,40],  [165,255,210]),
        "notes":"Safe for humans. Toxic to dogs.",
    },
    "strawberry":   {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([0,150,100],  [10,255,255]),
        "unripe_hsv": ([35,60,80],   [85,255,255]),
        "skin":       ([0,130,100],  [12,255,255]),
        "notes":"Safe raw. White or green ones are very sour.",
    },
    "watermelon":   {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([0,150,100],  [10,255,255]),
        "unripe_hsv": ([35,50,80],   [85,200,200]),
        "skin":       ([35,60,80],   [85,200,180]),   # green exterior
        "notes":"Safe raw. Rind edible when cooked.",
    },
    "pineapple":    {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([20,100,150], [35,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([22,100,160], [35,255,255]),
        "notes":"Safe. Unripe can cause stomach upset.",
    },
    "papaya":       {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([10,120,150], [25,255,255]),
        "unripe_hsv": ([35,60,100],  [85,200,255]),
        "skin":       ([12,110,160], [28,240,255]),
        "notes":"Safe ripe. Unripe contains latex — avoid if pregnant.",
    },
    "lemon":        {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([22,120,180], [35,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([22,110,190], [33,255,255]),
        "notes":"Safe. Very acidic — can erode tooth enamel.",
    },
    "pomegranate":  {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([0,120,80],   [15,255,200]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([0,100,80],   [15,255,200]),
        "notes":"Safe. Eat seeds (arils) only — discard rind.",
    },
    "kiwi":         {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([25,60,100],  [40,180,200]),
        "unripe_hsv": ([35,80,100],  [85,255,255]),
        "skin":       ([20,40,60],   [40,160,180]),
        "notes":"Safe including skin. High in vitamin C and K.",
    },
    "peach":        {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([8,100,180],  [20,220,255]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([8,80,190],   [20,200,255]),
        "notes":"Safe. Pit contains amygdalin — always discard it.",
    },
    "pear":         {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([22,60,150],  [40,200,255]),
        "unripe_hsv": ([35,80,80],   [85,255,200]),
        "skin":       ([25,50,160],  [40,200,255]),
        "notes":"Safe raw. High in fibre.",
    },
    "cherry":       {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([0,150,60],   [10,255,180]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([0,130,60],   [10,255,180]),
        "notes":"Safe. Never chew the pit — contains cyanogenic compounds.",
    },
    "coconut":      {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([15,30,60],   [30,120,180]),
        "unripe_hsv": ([35,60,100],  [85,200,255]),
        "skin":       ([15,20,50],   [30,100,170]),
        "notes":"Safe. Young coconut water is very nutritious.",
    },
    "guava":        {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([25,60,150],  [45,200,255]),
        "unripe_hsv": ([35,80,80],   [85,255,200]),
        "skin":       ([28,50,160],  [45,190,255]),
        "notes":"Safe raw including skin and seeds.",
    },
    "plum":         {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([130,60,60],  [160,255,200]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([128,50,50],  [162,255,200]),
        "notes":"Safe. Pit is toxic — always discard it.",
    },
    "fig":          {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([130,40,60],  [160,200,180]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([128,30,55],  [160,200,180]),
        "notes":"Safe. Milky sap in unripe figs irritates skin.",
    },
    "avocado":      {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([35,30,30],   [85,150,120]),
        "unripe_hsv": ([35,80,100],  [85,255,200]),
        "skin":       ([35,20,25],   [85,150,120]),
        "notes":"Safe for humans. Toxic to birds and most pets.",
    },
    "lime":         {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([35,100,100], [85,255,255]),
        "unripe_hsv": ([35,80,150],  [85,255,255]),
        "skin":       ([38,90,110],  [80,255,255]),
        "notes":"Safe. Great source of vitamin C.",
    },
    "lychee":       {
        "type":"Fruit", "safe":True,
        "ripe_hsv":   ([0,80,150],   [15,200,255]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([0,70,150],   [15,200,255]),
        "notes":"Safe. Remove skin and seed before eating.",
    },
    # ── VEGETABLES ──────────────────────────────────────────────────────────
    "broccoli":     {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,80,50],   [85,255,200]),
        "unripe_hsv": ([35,60,150],  [85,200,255]),
        "skin":       ([35,80,50],   [85,255,200]),
        "notes":"Safe raw or cooked. Cooking improves digestibility.",
    },
    "carrot":       {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([8,150,150],  [20,255,255]),
        "unripe_hsv": ([8,80,200],   [20,180,255]),
        "skin":       ([8,140,160],  [20,255,255]),
        "notes":"Safe raw or cooked. Rich in beta-carotene.",
    },
    "tomato":       {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([0,150,100],  [10,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([0,140,100],  [12,255,255]),
        "notes":"Safe ripe. Unripe contains solanine — limit intake.",
    },
    "potato":       {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([15,30,100],  [30,120,200]),
        "unripe_hsv": ([35,40,80],   [85,150,180]),
        "skin":       ([15,25,100],  [30,120,200]),
        "notes":"Must be cooked. Green parts contain solanine — avoid.",
    },
    "corn":         {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([20,100,150], [35,255,255]),
        "unripe_hsv": ([35,60,180],  [85,180,255]),
        "skin":       ([22,100,160], [35,255,255]),
        "notes":"Safe raw or cooked.",
    },
    "onion":        {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([15,40,150],  [30,150,255]),
        "unripe_hsv": ([35,60,100],  [85,200,255]),
        "skin":       ([15,30,160],  [28,150,255]),
        "notes":"Safe for humans. Toxic to dogs and cats.",
    },
    "garlic":       {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([15,20,180],  [30,80,255]),
        "unripe_hsv": ([35,30,150],  [85,120,255]),
        "skin":       ([15,15,190],  [30,80,255]),
        "notes":"Safe for humans. Toxic to dogs and cats.",
    },
    "spinach":      {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,80,50],   [85,255,180]),
        "unripe_hsv": ([35,60,150],  [85,200,255]),
        "skin":       ([38,80,50],   [80,255,175]),
        "notes":"Safe raw or cooked. High in iron and oxalates.",
    },
    "cucumber":     {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,60,80],   [85,200,200]),
        "unripe_hsv": ([35,80,150],  [85,255,255]),
        "skin":       ([38,60,80],   [82,200,200]),
        "notes":"Safe raw or cooked. Very hydrating.",
    },
    "eggplant":     {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([125,60,40],  [155,255,180]),
        "unripe_hsv": ([35,40,80],   [125,150,180]),
        "skin":       ([122,55,40],  [158,255,180]),
        "notes":"Cook before eating. Raw contains solanine.",
    },
    "capsicum":     {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([0,150,100],  [15,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([0,140,100],  [15,255,255]),
        "notes":"Safe raw or cooked. Very high in vitamin C.",
    },
    "cabbage":      {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,40,80],   [85,180,200]),
        "unripe_hsv": ([35,60,150],  [85,255,255]),
        "skin":       ([38,40,80],   [82,180,200]),
        "notes":"Safe raw or cooked. May cause gas in large amounts.",
    },
    "cauliflower":  {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([15,10,200],  [35,60,255]),
        "unripe_hsv": ([35,30,180],  [85,120,255]),
        "skin":       ([15,8,210],   [33,60,255]),
        "notes":"Safe raw or cooked.",
    },
    "peas":         {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,80,80],   [85,200,200]),
        "unripe_hsv": ([35,60,150],  [85,255,255]),
        "skin":       ([38,80,80],   [82,200,200]),
        "notes":"Safe raw or cooked. High in protein.",
    },
    "mushroom":     {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([15,20,60],   [30,80,180]),
        "unripe_hsv": ([15,10,150],  [30,50,255]),
        "skin":       ([15,15,60],   [30,80,180]),
        "notes":"Only eat known edible varieties. Wild mushrooms can be deadly.",
    },
    "ginger":       {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([15,40,150],  [30,120,230]),
        "unripe_hsv": ([35,40,150],  [85,120,255]),
        "skin":       ([18,35,155],  [30,120,230]),
        "notes":"Safe raw or cooked. Anti-inflammatory properties.",
    },
    "radish":       {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([0,120,80],   [15,255,220]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([0,110,80],   [14,255,220]),
        "notes":"Safe raw or cooked. Peppery flavour when raw.",
    },
    "beetroot":     {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([160,80,40],  [180,255,180]),
        "unripe_hsv": ([35,40,80],   [85,150,180]),
        "skin":       ([158,75,40],  [180,255,180]),
        "notes":"Safe raw or cooked. May turn urine red — harmless.",
    },
    "sweet potato": {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([8,100,150],  [20,220,255]),
        "unripe_hsv": ([8,60,180],   [20,150,255]),
        "skin":       ([10,90,155],  [20,220,255]),
        "notes":"Best cooked. Raw is hard to digest.",
    },
    "bitter gourd": {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,80,80],   [85,200,200]),
        "unripe_hsv": ([35,80,130],  [85,255,255]),
        "skin":       ([38,80,80],   [82,200,200]),
        "notes":"Safe. Bright red seeds cause vomiting — always discard them.",
    },
    "bottle gourd": {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,60,80],   [85,180,200]),
        "unripe_hsv": ([35,80,120],  [85,255,255]),
        "skin":       ([38,60,80],   [82,180,200]),
        "notes":"Cook before eating. Bitter taste means toxic — discard it.",
    },
    "chili":        {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([0,150,100],  [15,255,255]),
        "unripe_hsv": ([35,80,80],   [85,255,255]),
        "skin":       ([0,140,100],  [14,255,255]),
        "notes":"Safe. Capsaicin irritates eyes and skin — use gloves.",
    },
    "drumstick":    {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,40,80],   [85,150,200]),
        "unripe_hsv": ([35,60,120],  [85,200,255]),
        "skin":       ([38,40,80],   [82,150,200]),
        "notes":"Safe cooked. Leaves, pods, and seeds are all edible.",
    },
    "pumpkin":      {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([8,140,140],  [20,255,255]),
        "unripe_hsv": ([35,60,80],   [85,200,200]),
        "skin":       ([10,130,140], [22,255,255]),
        "notes":"Safe cooked or raw. Seeds are also nutritious.",
    },
    "zucchini":     {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,60,60],   [85,200,200]),
        "unripe_hsv": ([35,80,120],  [85,255,255]),
        "skin":       ([38,60,60],   [82,200,200]),
        "notes":"Safe raw or cooked.",
    },
    "turnip":       {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([130,30,180], [160,120,255]),
        "unripe_hsv": ([15,20,200],  [30,80,255]),
        "skin":       ([132,30,180], [158,120,255]),
        "notes":"Safe raw or cooked. High in vitamin C.",
    },
    "okra":         {
        "type":"Vegetable", "safe":True,
        "ripe_hsv":   ([35,60,60],   [85,200,200]),
        "unripe_hsv": ([35,80,100],  [85,255,255]),
        "skin":       ([38,60,60],   [82,200,200]),
        "notes":"Safe cooked. Mucilaginous texture helps digestion.",
    },
}

# ── Voice aliases — maps alternate spoken names to canonical KB names ──────
VOICE_ALIASES = {
    "bell pepper": "capsicum", "green pepper": "capsicum", "red pepper": "capsicum",
    "yellow pepper": "capsicum", "pepper": "capsicum",
    "hot pepper": "chili", "red chili": "chili", "green chili": "chili",
    "hot chilli": "chili", "chilli": "chili",
    "yam": "sweet potato", "sweet yam": "sweet potato",
    "beet": "beetroot", "beets": "beetroot", "red beet": "beetroot",
    "zucchini": "zucchini", "courgette": "zucchini",
    "brinjal": "eggplant", "aubergine": "eggplant",
    "lady finger": "okra", "ladies finger": "okra",
    "moringa": "drumstick",
    "grapes": "grape", "cherries": "cherry", "strawberries": "strawberry",
    "mangoes": "mango", "mangos": "mango",
    "tomatoes": "tomato", "potatoes": "potato",
    "mushrooms": "mushroom", "onions": "onion",
    "carrots": "carrot", "lemons": "lemon", "limes": "lime",
    "oranges": "orange", "apples": "apple", "bananas": "banana",
    "pineapples": "pineapple",
    # ── mode-switching phrases ─────────────────────────────────────────
    "camera mode": "__mode_camera__", "switch to camera": "__mode_camera__",
    "use camera": "__mode_camera__",
    "voice mode": "__mode_voice__",  "switch to voice": "__mode_voice__",
    "use voice": "__mode_voice__",
    "both mode": "__mode_both__",    "use both": "__mode_both__",
    "show all": "__mode_both__",
    # ── control phrases ───────────────────────────────────────────────
    "stop": "__stop__", "cancel": "__stop__", "clear": "__stop__",
    "reset": "__stop__", "never mind": "__stop__",
}

# YOLO COCO class names → KB names (extend as YOLO models improve)
COCO_TO_KB = {
    "apple":        "apple",
    "banana":       "banana",
    "orange":       "orange",
    "broccoli":     "broccoli",
    "carrot":       "carrot",
    "hot dog":      "chili",        # rough proxy
    "cake":         "pumpkin",      # rough proxy
}

# ── Operating modes ───────────────────────────────────────────────────────
MODES        = ["camera", "voice", "both"]
current_mode = {"value": "both"}


# ═══════════════════════════════════════════════════════════════
#  TTS ENGINE
# ═══════════════════════════════════════════════════════════════
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

# Pick a clear voice (prefer English)
voices = tts_engine.getProperty("voices")
for v in voices:
    if "english" in v.name.lower() or "en" in (v.languages[0].decode() if v.languages else "").lower():
        tts_engine.setProperty("voice", v.id)
        break

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
    """Speak text.  interrupt=True clears pending speech first."""
    if interrupt:
        with tts_queue.mutex:
            tts_queue.queue.clear()
    tts_queue.put(text)


def speak_detection(name, ripeness, source="camera", confirmed=False):
    """Build and speak a natural-language detection announcement."""
    info = PRODUCE_INFO.get(name, {})
    pfx  = "Voice and camera both confirm" if confirmed else \
           ("Camera detected" if source == "camera" else "Colour scan found")
    safe_phrase = "safe to eat" if info.get("safe") else "handle with caution"
    msg  = (f"{pfx}: {name}. "
            f"It appears {ripeness.lower()}. "
            f"It is {safe_phrase}. "
            f"{info.get('notes', '')}")
    speak(msg)


# ═══════════════════════════════════════════════════════════════
#  COLOUR-SIGNATURE CLASSIFIER
#  Works for every item in PRODUCE_INFO regardless of YOLO support
# ═══════════════════════════════════════════════════════════════
def classify_by_colour(roi_bgr, threshold=1500):
    """
    Scan an ROI (or whole frame) against every produce's skin HSV range.
    Returns list of (name, pixel_count) sorted best→worst.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return []
    hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
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
    """Return 'Ripe', 'Unripe / Raw', or 'Hard to tell'."""
    if name not in PRODUCE_INFO or roi_bgr is None or roi_bgr.size == 0:
        return "Hard to tell"
    info = PRODUCE_INFO[name]
    hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    rp   = int(np.sum(cv2.inRange(hsv,
               np.array(info["ripe_hsv"][0]),
               np.array(info["ripe_hsv"][1])) > 0))
    up   = int(np.sum(cv2.inRange(hsv,
               np.array(info["unripe_hsv"][0]),
               np.array(info["unripe_hsv"][1])) > 0))
    if rp > up and rp > 300:
        return "Ripe"
    if up > 300:
        return "Unripe / Raw"
    return "Hard to tell"


# ═══════════════════════════════════════════════════════════════
#  VOICE RECOGNISER  (background thread)
# ═══════════════════════════════════════════════════════════════
voice_state = {
    "text":      "",      # last raw transcript
    "active":    False,   # new unprocessed event?
    "target":    None,    # canonical produce name the user asked for
    "mode_cmd":  None,    # mode-switch command if any
    "stop_cmd":  False,   # user said stop/reset
}

recognizer                         = sr.Recognizer()
recognizer.energy_threshold        = 300
recognizer.dynamic_energy_threshold= True
mic_available                      = True


def resolve_spoken(text):
    """
    Parse a transcript into (produce_name | None, mode_cmd | None, stop | bool).
    Uses aliases first, then fuzzy substring matching against all KB names.
    """
    t = text.lower().strip()

    # 1) Alias table (includes mode commands and stop)
    for alias, val in VOICE_ALIASES.items():
        if alias in t:
            if val.startswith("__mode_"):
                return None, val, False
            if val == "__stop__":
                return None, None, True
            return val, None, False

    # 2) Direct produce name substring match
    for name in PRODUCE_INFO:
        if name in t:
            return name, None, False

    # 3) Partial / fuzzy match (longest matching substring)
    best, best_len = None, 0
    for name in PRODUCE_INFO:
        words = name.split()
        for w in words:
            if len(w) >= 4 and w in t and len(w) > best_len:
                best     = name
                best_len = len(w)
    if best:
        return best, None, False

    return None, None, False


def listen_loop():
    global mic_available
    try:
        mic = sr.Microphone()
    except OSError:
        mic_available = False
        voice_state["text"] = "No microphone found"
        return

    speak("Fruit and vegetable detector ready. "
          "Show produce to the camera, say its name, or do both together.")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            text = recognizer.recognize_google(audio).lower()
            voice_state["text"]   = text
            voice_state["active"] = True

            produce, mode_cmd, stop = resolve_spoken(text)

            voice_state["target"]   = produce
            voice_state["mode_cmd"] = mode_cmd
            voice_state["stop_cmd"] = stop

            if stop:
                voice_state["target"] = None
                speak("Target cleared.")
            elif mode_cmd:
                new_mode = mode_cmd.replace("__mode_", "").replace("__", "")
                current_mode["value"] = new_mode
                speak(f"Switched to {new_mode} mode.")
            elif produce:
                speak(f"Looking for {produce} on camera now.", interrupt=True)
            else:
                speak("I did not catch a produce name. Please try again.")

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
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════
FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_size_label(box_area, frame_area):
    r = box_area / max(frame_area, 1)
    if r > 0.30: return "Large"
    if r > 0.10: return "Medium"
    return "Small"


def draw_panel(frame, x1, y1, x2, y2, name, ripeness, size, info, conf,
               source="camera", voice_confirmed=False):
    """Bounding box + floating info panel."""
    if voice_confirmed:
        box_col, thick = (0, 255, 255), 3
    elif source == "voice":
        box_col, thick = (0, 180, 255), 2
    elif info["safe"]:
        box_col, thick = (0, 220, 100), 2
    else:
        box_col, thick = (0, 60, 220), 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, thick)
    if voice_confirmed:
        cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 0), 1)

    ph = 155
    px = max(x1, 0)
    py = max(y1 - ph - 2, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + 330, py + ph), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    src_badge = "[VOICE+CAM]" if voice_confirmed else \
                ("[COLOUR SCAN]" if source == "colour" else
                 ("[VOICE]" if source == "voice" else "[CAMERA]"))
    safe_str   = "SAFE TO EAT" if info["safe"] else "CAUTION"
    safe_color = (80, 255, 130) if info["safe"] else (80, 110, 255)

    lines = [
        (f"{name.upper()}  {src_badge}",       (255, 255, 255), 0.54, 1),
        (f"Type     : {info['type']}",          (200, 200, 200), 0.44, 1),
        (f"Ripeness : {ripeness}",              (100, 220, 255), 0.44, 1),
        (f"Size     : {size}",                  (200, 200, 200), 0.44, 1),
        (f"Safety   : {safe_str}",              safe_color,      0.44, 1),
        (f"{info['notes'][:50]}",               (170, 170, 170), 0.37, 1),
    ]
    tx, ty = px + 8, py + 22
    for text, color, scale, thick_l in lines:
        cv2.putText(frame, text, (tx, ty), FONT, scale, color, thick_l, cv2.LINE_AA)
        ty += 24

    cv2.putText(frame, f"{conf:.0%}", (x2 - 52, y2 - 6),
                FONT, 0.48, (255, 255, 160), 1, cv2.LINE_AA)


def draw_voice_overlay(frame, target_name, ripeness, found=True):
    """Full-frame voice-mode overlay when colour scan picks up the target."""
    info = PRODUCE_INFO.get(target_name, {})
    h, w = frame.shape[:2]

    # tinted border
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 160, 255), -1)
    cv2.addWeighted(ov, 0.06, frame, 0.94, 0, frame)
    cv2.rectangle(frame, (14, 14), (w - 14, h - 14), (0, 200, 255), 2)

    safe_str   = "SAFE TO EAT" if info.get("safe") else "CAUTION"
    safe_color = (0, 255, 130) if info.get("safe") else (0, 80, 255)

    lines = [
        (f"VOICE TARGET : {target_name.upper()}",          (0, 220, 255),  0.68, 2),
        (f"Type         : {info.get('type', '?')}",        (220, 220, 220),0.52, 1),
        (f"Ripeness     : {ripeness}",                     (100, 220, 255),0.52, 1),
        (f"Safety       : {safe_str}",                     safe_color,     0.52, 1),
        (f"Note : {info.get('notes', '')[:55]}",           (180, 180, 180),0.44, 1),
    ]
    y = 70
    for text, color, scale, thick_l in lines:
        cv2.putText(frame, text, (30, y), FONT, scale, color, thick_l, cv2.LINE_AA)
        y += 34

    if not found:
        cv2.putText(frame, "Hold item closer — colour scan inconclusive",
                    (30, y + 10), FONT, 0.46, (0, 130, 255), 1, cv2.LINE_AA)


def draw_colour_hits(frame, hits, frame_area):
    """Draw small badges for colour-scan hits that YOLO missed."""
    h, w = frame.shape[:2]
    x_off = 10
    for name, cnt in hits[:4]:
        info     = PRODUCE_INFO[name]
        ripeness = estimate_ripeness(frame, name)
        safe_str = "SAFE" if info["safe"] else "CAUTION"
        badge    = f"{name.upper()}  [{safe_str}]  {ripeness}"
        bw, bh   = cv2.getTextSize(badge, FONT, 0.46, 1)[0]
        by       = h - 56     # above HUD bar
        cv2.rectangle(frame, (x_off - 4, by - bh - 6),
                      (x_off + bw + 4, by + 4), (20, 20, 20), -1)
        cv2.putText(frame, badge, (x_off, by),
                    FONT, 0.46,
                    (0, 255, 130) if info["safe"] else (0, 80, 255),
                    1, cv2.LINE_AA)
        x_off += bw + 20


def draw_hud(frame, mode, heard, target):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 44), (w, h), (12, 12, 12), -1)
    mic_icon = "MIC ON" if mic_available else "NO MIC"
    heard_t  = (heard or "Listening...")[:55]
    target_t = target or "—"
    cv2.putText(frame,
                f"[{mic_icon}]  Heard: {heard_t}   |   Target: {target_t}   |   Q=quit  M=mode",
                (10, h - 14), FONT, 0.44, (160, 220, 160), 1, cv2.LINE_AA)

    mode_colors = {"camera": (80, 200, 80), "voice": (80, 180, 255), "both": (0, 220, 255)}
    mc    = mode_colors.get(mode, (200, 200, 200))
    badge = f"MODE: {mode.upper()}"
    bw    = cv2.getTextSize(badge, FONT, 0.52, 1)[0][0]
    cv2.rectangle(frame, (w - bw - 22, 8), (w - 8, 34), (20, 20, 20), -1)
    cv2.putText(frame, badge, (w - bw - 14, 27), FONT, 0.52, mc, 1, cv2.LINE_AA)


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
speak_cooldown  = 0.0

# Cooldown tracker per-target for voice requests
voice_last_spoken = {}

print("=" * 70)
print("  FRUIT & VEGETABLE DETECTOR  |  Voice + Camera  |  40+ Items")
print("  Q = quit     M = cycle mode (camera / voice / both)")
print("  Voice commands:")
print("    Say any fruit or vegetable name to target it")
print("    'camera mode' / 'voice mode' / 'both mode'")
print("    'stop' or 'reset' to clear target")
print("=" * 70)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    h, w       = frame.shape[:2]
    frame_area = h * w
    mode       = current_mode["value"]
    now        = time.time()

    yolo_found  = {}   # name → (x1, y1, x2, y2, ripeness, size, conf)
    colour_hits = []   # from frame-level colour scan

    # ── LAYER 1 : YOLO detection ────────────────────────────────────────
    if mode in ("camera", "both"):
        results = model(frame, conf=0.40, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                if label not in COCO_TO_KB:
                    continue
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(x1, 0), max(y1, 0)
                roi      = frame[y1:y2, x1:x2]
                kb_name  = COCO_TO_KB[label]
                ripeness = estimate_ripeness(roi, kb_name)
                size     = get_size_label((x2 - x1) * (y2 - y1), frame_area)
                yolo_found[kb_name] = (x1, y1, x2, y2, ripeness, size, conf)

    # ── LAYER 2 : Frame-level colour scan (always, for items YOLO misses) ─
    voice_target = voice_state["target"]

    # Targeted scan — only for voice target if YOLO missed it
    if voice_target and voice_target not in yolo_found:
        colour_hits = classify_by_colour(frame, threshold=1500)
        colour_found = {n: cnt for n, cnt in colour_hits}
    else:
        # Passive broad scan — top 3 best-matching items
        colour_hits  = classify_by_colour(frame, threshold=2500)
        colour_found = {n: cnt for n, cnt in colour_hits}

    # ── DRAW YOLO detections ────────────────────────────────────────────
    for kb_name, vals in yolo_found.items():
        x1, y1, x2, y2, ripeness, size, conf = vals
        info      = PRODUCE_INFO[kb_name]
        confirmed = (voice_target == kb_name)

        draw_panel(frame, x1, y1, x2, y2, kb_name, ripeness, size,
                   info, conf, source="camera", voice_confirmed=confirmed)

        key = f"yolo-{kb_name}-{ripeness}"
        if key != last_spoken_key and now > speak_cooldown:
            speak_detection(kb_name, ripeness,
                            source="camera", confirmed=confirmed)
            last_spoken_key = key
            speak_cooldown  = now + 7

    # ── VOICE INPUT EVENT ───────────────────────────────────────────────
    if voice_state["active"]:
        voice_state["active"] = False

        if voice_state["stop_cmd"]:
            voice_state["stop_cmd"] = False

        elif voice_target and mode in ("voice", "both"):
            if voice_target not in yolo_found:
                # YOLO missed it — rely on colour scan result
                if voice_target in colour_found:
                    ripeness = estimate_ripeness(frame, voice_target)
                    key = f"colour-{voice_target}-{ripeness}"
                    if key != last_spoken_key:
                        speak_detection(voice_target, ripeness,
                                        source="colour", confirmed=False)
                        last_spoken_key = key
                        speak_cooldown  = now + 7
                else:
                    speak(f"Cannot clearly identify {voice_target} in the frame. "
                          f"Hold it up closer and try again.")

    # ── VOICE-ONLY MODE : continuous target scan ─────────────────────────
    if mode == "voice" and voice_target:
        if voice_target in yolo_found:
            x1, y1, x2, y2, ripeness, size, conf = yolo_found[voice_target]
            info = PRODUCE_INFO[voice_target]
            draw_panel(frame, x1, y1, x2, y2, voice_target, ripeness, size,
                       info, conf, source="camera", voice_confirmed=True)
        else:
            ripeness = estimate_ripeness(frame, voice_target)
            found    = voice_target in colour_found
            draw_voice_overlay(frame, voice_target, ripeness, found=found)

            # periodic re-announcement in voice mode
            key = f"voice-{voice_target}-{ripeness}"
            if key != last_spoken_key and now > speak_cooldown:
                if found:
                    speak_detection(voice_target, ripeness,
                                    source="colour", confirmed=False)
                    last_spoken_key = key
                    speak_cooldown  = now + 8

    # ── PASSIVE COLOUR BADGES (camera / both mode) ────────────────────────
    if mode in ("camera", "both"):
        # Show colour hits that YOLO did not already catch
        extra = [(n, c) for n, c in colour_hits if n not in yolo_found]
        if extra:
            draw_colour_hits(frame, extra, frame_area)

            # Announce top hit if not already speaking something
            top_name, _ = extra[0]
            ripeness_top = estimate_ripeness(frame, top_name)
            key = f"colour-{top_name}-{ripeness_top}"
            if key != last_spoken_key and now > speak_cooldown:
                speak_detection(top_name, ripeness_top,
                                source="colour", confirmed=(top_name == voice_target))
                last_spoken_key = key
                speak_cooldown  = now + 7

    # ── HUD ──────────────────────────────────────────────────────────────
    draw_hud(frame, mode, voice_state["text"], voice_target)

    cv2.imshow("Fruit & Vegetable Detector  |  Voice + Camera  |  40+ Items", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        idx = MODES.index(current_mode["value"])
        current_mode["value"] = MODES[(idx + 1) % len(MODES)]
        speak(f"Mode switched to {current_mode['value']}.")

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)   # signal TTS thread to exit cleanly