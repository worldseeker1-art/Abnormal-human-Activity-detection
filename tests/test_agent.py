import time

import numpy as np

from app.agent import SurveillanceAgent
from app.detectors import CrowdDetector, FaceLoiterDetector, TamperDetector
from app.wiring import Scene


def _frame(color: int = 80) -> np.ndarray:
    return np.full((180, 320, 3), color, dtype=np.uint8)


def test_loiter_fires_once_after_dwell():
    det = FaceLoiterDetector(loiter_seconds=2)
    boxes = [[10, 10, 40, 40]]
    t0 = 1_000.0
    early = det.process(_frame()[:, :, 0], t0, boxes=boxes)
    late = det.process(_frame()[:, :, 0], t0 + 2.1, boxes=boxes)
    again = det.process(_frame()[:, :, 0], t0 + 3.0, boxes=boxes)
    assert early.active is False
    assert late.active is True
    assert again.active is False


def test_empty_bay_resets_loiter():
    det = FaceLoiterDetector(loiter_seconds=1)
    det.process(_frame()[:, :, 0], 10.0, boxes=[[1, 1, 2, 2]])
    empty = det.process(_frame()[:, :, 0], 12.0, boxes=[])
    assert empty.count == 0
    assert det.present_since is None


def test_crowd_only_fires_once_per_wave():
    det = CrowdDetector(max_faces=2)
    assert det.process(4).active is True
    assert det.process(4).active is False
    assert det.process(1).active is False
    assert det.process(5).active is True


def test_tamper_forced_and_dark():
    det = TamperDetector()
    mid = np.full((90, 160), 80, dtype=np.uint8)
    dark = np.zeros((90, 160), dtype=np.uint8)
    det.process(mid)
    assert det.process(dark).active is True
    det2 = TamperDetector()
    det2.process(mid)
    assert det2.process(mid, forced=True).active is True


def test_agent_emits_crowd_and_loiter(tmp_path, monkeypatch):
    agent = SurveillanceAgent(loiter_seconds=1)
    agent.alerts.log_path = tmp_path / "alerts.jsonl"
    scene = Scene(frame=_frame(), people=[[10, 10, 40, 80], [60, 10, 40, 80], [110, 10, 40, 80]], source="synthetic")
    first = agent.ingest(scene, now=time.time())
    assert first["crowd"]["active"] is True
    later = agent.ingest(scene, now=time.time() + 1.2)
    assert later["face"]["active"] is True
    kinds = {a["kind"] for a in agent.alerts.recent()}
    assert "crowd" in kinds
    assert "face_loiter" in kinds
    agent.stop()
