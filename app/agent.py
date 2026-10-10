"""Surveillance agent: scores each scene and raises alerts once per wave."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from .detectors import CrowdDetector, FaceLoiterDetector, MotionDetector, TamperDetector
from .wiring import AlertBus, CameraSource, Scene


class SurveillanceAgent:
    def __init__(self, loiter_seconds: float = 120.0) -> None:
        self.camera = CameraSource()
        self.alerts = AlertBus()
        self.face = FaceLoiterDetector(loiter_seconds=loiter_seconds)
        self.crowd = CrowdDetector()
        self.motion = MotionDetector()
        self.tamper = TamperDetector()
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._status: dict = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.started_at = time.time()
        self.frames = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.camera.close()

    def snapshot(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        with self._lock:
            body = dict(self._status)
        body["running"] = self._running
        body["camera"] = self.camera.mode
        body["uptime_s"] = int(time.time() - self.started_at)
        body["frames"] = self.frames
        body["alerts"] = self.alerts.recent(40)
        return body

    def ingest(self, scene: Scene, now: Optional[float] = None) -> dict:
        """Score one scene. Used by the live loop and tests."""
        now = time.time() if now is None else now
        gray = cv2.cvtColor(scene.frame, cv2.COLOR_BGR2GRAY)
        known = scene.people if scene.source == "synthetic" else None
        face = self.face.process(gray, now, boxes=known)
        crowd = self.crowd.process(face.count)
        motion = self.motion.process(gray)
        tamper = self.tamper.process(gray, forced=scene.tamper)

        if face.active:
            self.alerts.emit("face_loiter", f"Loitering at ATM: {face.detail}")
        if crowd.active:
            self.alerts.emit("crowd", f"Crowd at ATM: {crowd.detail}")
        if tamper.active:
            self.alerts.emit("tamper", f"Camera tamper: {tamper.detail}", severity="high")

        vis = scene.frame.copy()
        for (x, y, w, h) in face.boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (220, 160, 40), 2)
        cv2.putText(
            vis,
            f"{face.detail}  |  {tamper.detail}",
            (16, vis.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        status = {
            "face": {"count": face.count, "detail": face.detail, "active": face.active},
            "crowd": {"count": crowd.count, "detail": crowd.detail, "active": crowd.active},
            "motion": {"detail": motion.detail, "active": motion.active},
            "tamper": {"detail": tamper.detail, "active": tamper.active},
        }
        with self._lock:
            if ok:
                self._jpeg = buf.tobytes()
            self._status = status
        return status

    def _loop(self) -> None:
        while self._running:
            scene = self.camera.read()
            if scene is None:
                time.sleep(0.05)
                continue
            self.frames += 1
            self.ingest(scene)
            time.sleep(0.04)
