from pathlib import Path

import cv2
import numpy as np

import detector as detector_module
from domain import BoundingBox
from detector import PersonDetector


def _make_test_video(path: Path, frame_count: int = 20, fps: float = 5.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for idx in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = min(255, idx * 10)
        writer.write(frame)
    writer.release()
    return path


def test_tracker_keeps_single_track_for_stable_motion(tmp_path: Path, monkeypatch):
    video_path = _make_test_video(tmp_path / "stable.mp4", frame_count=16, fps=5.0)
    detector = PersonDetector("fake.pt")

    monkeypatch.setattr(detector_module, "get_model", lambda model_path: object())

    def fake_detect_frame(model, frame):
        idx = int(round(float(frame[:, :, 1].mean()) / 10.0))
        x1 = 4 + idx
        return [BoundingBox(x1=x1, y1=8, x2=x1 + 10, y2=24, confidence=0.9)]

    monkeypatch.setattr(detector, "detect_frame", fake_detect_frame)

    result = detector.process_range(
        str(video_path),
        0.0,
        3.0,
        0.75,
        stable_person_interval=0.75,
        refined_sample_interval=0.25,
        strong_refined_sample_interval=0.1,
    )

    confirmed_track_ids = {track_id for frame in result.detections for track_id in frame.track_ids}
    assert confirmed_track_ids == {1}
    assert result.track_summary[0]["track_id"] == 1


def test_adaptive_sampling_refines_near_person_count_change(tmp_path: Path, monkeypatch):
    video_path = _make_test_video(tmp_path / "adaptive.mp4", frame_count=20, fps=5.0)
    detector = PersonDetector("fake.pt")

    monkeypatch.setattr(detector_module, "get_model", lambda model_path: object())

    def fake_detect_frame(model, frame):
        idx = int(round(float(frame[:, :, 1].mean()) / 10.0))
        if idx < 6:
            return [BoundingBox(x1=8, y1=8, x2=18, y2=24, confidence=0.9)]
        return [
            BoundingBox(x1=8, y1=8, x2=18, y2=24, confidence=0.9),
            BoundingBox(x1=34, y1=8, x2=46, y2=26, confidence=0.88),
        ]

    monkeypatch.setattr(detector, "detect_frame", fake_detect_frame)

    result = detector.process_range(
        str(video_path),
        0.0,
        3.5,
        1.0,
        stable_person_interval=0.75,
        refined_sample_interval=0.25,
        strong_refined_sample_interval=0.1,
    )

    refined_frames = [frame for frame in result.detections if frame.sampling_mode == "refined"]
    assert refined_frames
    assert result.sampling_profile["sampling_events"]
    assert any(0.9 <= frame.timestamp <= 1.4 for frame in refined_frames)
