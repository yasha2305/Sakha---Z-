"""
Automatic spray bottle controller
---------------------------------
- Sprays on a fixed time interval.
- Sprays again after each displacement: whenever the bottle moves farther than
  a set threshold from the last position (or when you report a displacement delta).

Replace `_pulse_hw()` with your pump/solenoid driver (e.g. Raspberry Pi GPIO).

Usage (demo simulation):
    python automatic_spray_bottle.py

Programmatic:
    from automatic_spray_bottle import AutomaticSprayBottle, SprayConfig
    bottle = AutomaticSprayBottle(SprayConfig(interval_seconds=4.0))
    bottle.start()
    bottle.update_position(0.1, 0.0)  # movement → extra spray
    bottle.stop()
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SprayConfig:
    """Tuning for interval + displacement behaviour."""

    interval_seconds: float = 5.0
    """Spray automatically every this many seconds while running."""

    displacement_threshold: float = 0.05
    """Minimum movement (same units as x,y,z) to count as one displacement event."""

    min_seconds_between_displacement_sprays: float = 0.8
    """Avoid spraying too often if the bottle jitters near the threshold."""

    spray_duration_seconds: float = 0.25
    """How long the actuator stays on (for real hardware; demo logs only)."""


class AutomaticSprayBottle:
    def __init__(
        self,
        config: SprayConfig,
        on_spray: Optional[Callable[[], None]] = None,
    ):
        self.config = config
        self._on_spray = on_spray or self._default_on_spray
        self._lock = threading.Lock()
        self._running = False
        self._interval_thread: Optional[threading.Thread] = None
        self._last_position: Optional[tuple[float, float, float]] = None
        self._last_displacement_spray_mono: float = 0.0

    @staticmethod
    def _default_on_spray() -> None:
        print(f"[SPRAY] {time.strftime('%H:%M:%S')} — liquid dispensed")

    def _pulse_hw(self) -> None:
        """
        Hook for hardware: drive pump/solenoid HIGH, sleep spray_duration, then LOW.
        Demo only calls on_spray().
        """
        self._on_spray()
        # Example for Raspberry Pi (uncomment & install RPi.GPIO):
        # GPIO.output(PUMP_PIN, GPIO.HIGH)
        # time.sleep(self.config.spray_duration_seconds)
        # GPIO.output(PUMP_PIN, GPIO.LOW)

    def spray_now(self) -> None:
        """Force one spray (thread-safe)."""
        with self._lock:
            self._pulse_hw()

    def update_position(self, x: float, y: float, z: float = 0.0) -> None:
        """
        Call whenever you have a new bottle position (meters, cm, or arbitrary units).
        If distance from last position >= displacement_threshold, triggers a spray.
        """
        now = time.monotonic()
        with self._lock:
            if self._last_position is None:
                self._last_position = (x, y, z)
                return
            ox, oy, oz = self._last_position
            dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2)
            self._last_position = (x, y, z)
            if dist < self.config.displacement_threshold:
                return
            if now - self._last_displacement_spray_mono < self.config.min_seconds_between_displacement_sprays:
                return
            self._last_displacement_spray_mono = now
        self._pulse_hw()

    def notify_displacement(self, distance: float) -> None:
        """
        If you only know how far the bottle moved this step (e.g. encoder delta),
        call this instead of update_position. Sprays when distance >= threshold.
        """
        if distance < self.config.displacement_threshold:
            return
        now = time.monotonic()
        with self._lock:
            if now - self._last_displacement_spray_mono < self.config.min_seconds_between_displacement_sprays:
                return
            self._last_displacement_spray_mono = now
        self._pulse_hw()

    def _interval_loop(self) -> None:
        while self._running:
            time.sleep(self.config.interval_seconds)
            if not self._running:
                break
            self._pulse_hw()

    def start(self) -> None:
        """Begin periodic spraying and start listening for displacement updates."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._interval_thread = threading.Thread(target=self._interval_loop, daemon=True)
        self._interval_thread.start()

    def stop(self) -> None:
        """Stop the interval loop."""
        self._running = False
        if self._interval_thread is not None:
            self._interval_thread.join(timeout=self.config.interval_seconds + 1.0)
            self._interval_thread = None


def _demo() -> None:
    cfg = SprayConfig(
        interval_seconds=3.0,
        displacement_threshold=0.02,
        min_seconds_between_displacement_sprays=0.5,
    )
    bottle = AutomaticSprayBottle(cfg)
    print("Demo: interval spray every 3s; movement > 0.02 triggers extra spray.")
    print("Simulating bottle path for 12 seconds…\n")
    bottle.start()
    bottle.update_position(0.0, 0.0)
    t0 = time.monotonic()
    path = [
        (0.0, 0.0),
        (0.01, 0.0),
        (0.05, 0.0),
        (0.05, 0.04),
        (0.12, 0.08),
    ]
    i = 0
    while time.monotonic() - t0 < 12.0:
        time.sleep(0.9)
        if i < len(path):
            x, y = path[i]
            bottle.update_position(x, y)
            i += 1
    bottle.stop()
    print("\nDemo finished.")


if __name__ == "__main__":
    _demo()
