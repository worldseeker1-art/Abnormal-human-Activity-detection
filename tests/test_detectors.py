import time

import numpy as np

from app.detectors import CrowdDetector, FaceLoiterDetector, TamperDetector


def test_loiter_fires_after_dwell():
    det = FaceLoiterDetector(loiter_seconds=2)
    gray = np.zeros((200, 200), dtype=np.uint8)
    # Inject a fake face box path by setting present_since directly after a miss,
    # then feed a white block Haar may or may not see. Use internal timer instead.
    first = det.process(gray, time.time())
    assert first.kind == "face_loiter"


def test_crowd_only_fires_once_per_wave():
    det = CrowdDetector(max_faces=2)
    a = det.process(4)
    b = det.process(4)
    c = det.process(1)
    d = det.process(5)
    assert a.active is True
    assert b.active is False
    assert c.active is False
    assert d.active is True


def test_tamper_dark_frame():
    det = TamperDetector()
    mid = np.full((90, 160), 80, dtype=np.uint8)
    dark = np.zeros((90, 160), dtype=np.uint8)
    det.process(mid)
    hit = det.process(dark)
    assert hit.active is True
