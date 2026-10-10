"""Face/loiter, crowd, motion, and tamper detectors."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FACE_XML = ROOT / "haarcascade_frontalface_default.xml"


@dataclass
class Detection:
    kind: str
    count: int = 0
    active: bool = False
    detail: str = ""
    boxes: list = field(default_factory=list)


class FaceLoiterDetector:
    def __init__(self, loiter_seconds: float = 120.0) -> None:
        path = str(FACE_XML) if FACE_XML.exists() else os.getenv("FACE_CASCADE", "")
        self.cascade = cv2.CascadeClassifier(path) if path else cv2.CascadeClassifier()
        self.loiter_seconds = loiter_seconds
        self.present_since: Optional[float] = None
        self.fired = False

    def process(
        self,
        gray: np.ndarray,
        now: float,
        boxes: Optional[list[list[int]]] = None,
    ) -> Detection:
        if boxes is None:
            faces = []
            if not self.cascade.empty():
                faces = [list(map(int, box)) for box in self.cascade.detectMultiScale(gray, 1.3, 5)]
        else:
            faces = boxes
        if faces:
            if self.present_since is None:
                self.present_since = now
                self.fired = False
            dwell = now - self.present_since
            loitering = dwell >= self.loiter_seconds and not self.fired
            if loitering:
                self.fired = True
            return Detection(
                kind="face_loiter",
                count=len(faces),
                active=loitering,
                detail=f"{len(faces)} person(s), {int(dwell)}s in bay",
                boxes=faces,
            )
        self.present_since = None
        self.fired = False
        return Detection(kind="face_loiter", detail="bay empty")


class CrowdDetector:
    def __init__(self, max_faces: int = 2) -> None:
        self.max_faces = max_faces
        self.fired = False

    def process(self, face_count: int) -> Detection:
        crowded = face_count > self.max_faces
        first = crowded and not self.fired
        self.fired = crowded
        return Detection(
            kind="crowd",
            count=face_count,
            active=first,
            detail=f"{face_count} people in bay (limit {self.max_faces})",
        )


class MotionDetector:
    def __init__(self) -> None:
        self._prev: Optional[np.ndarray] = None

    def process(self, gray: np.ndarray) -> Detection:
        small = cv2.resize(gray, (160, 90))
        if self._prev is None:
            self._prev = small
            return Detection(kind="motion", detail="warming up")
        delta = cv2.absdiff(self._prev, small)
        self._prev = small
        score = float(delta.mean())
        return Detection(
            kind="motion",
            count=int(score),
            active=score > 8.0,
            detail=f"motion score {score:.1f}",
        )


class TamperDetector:
    def __init__(self) -> None:
        self._prev: Optional[np.ndarray] = None
        self.fired = False

    def process(self, gray: np.ndarray, forced: bool = False) -> Detection:
        small = cv2.resize(gray, (160, 90))
        mean = float(small.mean())
        if self._prev is None:
            self._prev = small
            if not forced:
                return Detection(kind="tamper", detail="warming up")
        delta = float(cv2.absdiff(self._prev, small).mean()) if self._prev is not None else 0.0
        self._prev = small
        dark = mean < 12
        smash = delta > 45
        hit = forced or dark or smash
        first = hit and not self.fired
        self.fired = hit
        if forced or dark:
            reason = "signal loss / covered lens"
        elif smash:
            reason = "scene smash"
        else:
            reason = "stable"
        return Detection(kind="tamper", active=first, detail=reason)
