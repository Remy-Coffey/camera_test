from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig
from services.app import create_app


@pytest.fixture()
def temp_config(tmp_path: Path) -> AppConfig:
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    return AppConfig(
        base_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        artifact_dir=tmp_path / "artifacts",
        frontend_dir=frontend_dir,
        database_path=tmp_path / "camera_test.sqlite3",
        detector_model_path=tmp_path / "fake-model.pt",
        max_workers=2,
        worker_mode="inline",
        llm_enabled=False,
        video_llm_enabled=False,
    )


@pytest.fixture()
def client(temp_config: AppConfig) -> TestClient:
    app = create_app(temp_config)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    for idx in range(10):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 20 * idx
        writer.write(frame)
    writer.release()
    return video_path
