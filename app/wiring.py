"""Camera input, scene packets, and alert output."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np


@dataclass
class Scene:
    frame: np.ndarray
    people: list[list[int]] = field(default_factory=list)
    tamper: bool = False
    source: str = "webcam"


@dataclass
class Alert:
    id: int
    kind: str
    severity: str
    message: str
    ts: float

    def to_dict(self) -> dict:
        return asdict(self)


class CameraSource:
    """Webcam if present. Otherwise a scripted ATM bay that the agent can score."""

    def __init__(self, device: int = 0, width: int = 960, height: int = 540) -> None:
        self.width = width
        self.height = height
        self._synthetic = False
        self._t0 = time.time()
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            self._synthetic = True
            self.cap.release()
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def mode(self) -> str:
        return "synthetic" if self._synthetic else "webcam"

    def read(self) -> Optional[Scene]:
        if self._synthetic:
            return self._synthetic_scene()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return self._synthetic_scene()
        return Scene(frame=frame, source="webcam")

    def _synthetic_scene(self) -> Scene:
        t = time.time() - self._t0
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (18, 16, 14)
        cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (46, 42, 38), 2)
        cv2.putText(
            frame,
            "ATM BAY 01",
            (56, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (210, 200, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (120, 220), (360, 480), (70, 62, 54), -1)
        cv2.rectangle(frame, (140, 250), (340, 340), (20, 220, 200), -1)
        cv2.putText(frame, "ATM", (210, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 20, 20), 2)

        cycle = int(t) % 45
        tamper = cycle >= 42
        people: list[list[int]] = []
        if not tamper:
            count = 1
            if 14 <= cycle < 28:
                count = 4
            elif cycle >= 28:
                count = 2
            cx = int(520 + 160 * np.sin(t * 0.55))
            cy = 300
            offsets = [0, -110, 110, 200][:count]
            for ox in offsets:
                px, py = cx + ox, cy
                cv2.ellipse(frame, (px, py), (46, 90), 0, 0, 360, (40, 40, 200), -1)
                cv2.circle(frame, (px, py - 110), 28, (50, 55, 210), -1)
                people.append([px - 46, py - 140, 92, 230])
        else:
            frame[:] = (4, 4, 4)
            cv2.putText(frame, "SIGNAL LOSS", (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3)

        cv2.putText(
            frame,
            time.strftime("%H:%M:%S"),
            (w - 180, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 220, 180),
            2,
        )
        return Scene(frame=frame, people=people, tamper=tamper, source="synthetic")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class AlertBus:
    """Memory + jsonl log, optional webhook / MSG91."""

    def __init__(self, max_items: int = 200, log_path: Optional[Path] = None) -> None:
        self._items: Deque[Alert] = deque(maxlen=max_items)
        self._lock = threading.Lock()
        self._next_id = 1
        self.log_path = log_path or Path("data/alerts.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        self.sms_auth = os.getenv("MSG91_AUTHKEY", "").strip()
        self.sms_to = os.getenv("ALERT_SMS_TO", "").strip()
        self.sms_sender = os.getenv("ALERT_SMS_SENDER", "ATMAUT")

    def emit(self, kind: str, message: str, severity: str = "high") -> Alert:
        with self._lock:
            alert = Alert(self._next_id, kind, severity, message, time.time())
            self._next_id += 1
            self._items.appendleft(alert)
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.to_dict()) + "\n")
        except OSError:
            pass
        self._fanout(alert)
        return alert

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [a.to_dict() for a, _ in zip(self._items, range(limit))]

    def _fanout(self, alert: Alert) -> None:
        if self.webhook_url:
            threading.Thread(target=self._post_webhook, args=(alert,), daemon=True).start()
        if self.sms_auth and self.sms_to:
            threading.Thread(target=self._send_sms, args=(alert,), daemon=True).start()

    def _post_webhook(self, alert: Alert) -> None:
        req = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(alert.to_dict()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, TimeoutError):
            return

    def _send_sms(self, alert: Alert) -> None:
        payload = json.dumps(
            {
                "sender": self.sms_sender,
                "route": "4",
                "country": "91",
                "sms": [{"message": alert.message, "to": [self.sms_to]}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.msg91.com/api/v2/sendsms",
            data=payload,
            headers={"authkey": self.sms_auth, "content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
        except (urllib.error.URLError, TimeoutError):
            return
