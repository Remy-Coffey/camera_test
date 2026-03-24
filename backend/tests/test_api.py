from __future__ import annotations

from pathlib import Path

from domain import ArtifactKind
import services.orchestrator as orchestrator_module


def test_upload_start_and_result_flow(client, sample_video: Path, monkeypatch):
    def fake_run_chunk(item):
        base = item.start_time
        return {
            "chunk_id": item.chunk_id,
            "task_id": item.task_id,
            "detections": [
                {
                    "frame_index": 0,
                    "timestamp": round(base, 3),
                    "persons": [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "confidence": 0.9}],
                    "person_count": 1,
                    "track_ids": [1],
                    "scene_change_score": 0.0,
                    "scene_changed": False,
                },
                {
                    "frame_index": 1,
                    "timestamp": round(base + 1, 3),
                    "persons": [{"x1": 4, "y1": 0, "x2": 14, "y2": 10, "confidence": 0.9}],
                    "person_count": 1,
                    "track_ids": [1],
                    "scene_change_score": 0.0,
                    "scene_changed": False,
                },
            ],
            "video_info": {"fps": 5.0, "total_frames": 10, "width": 64, "height": 48, "duration": 2.0},
            "frame_count": 2,
            "track_summary": [{"track_id": 1, "first_seen": round(base, 3), "last_seen": round(base + 1, 3), "frame_count": 2}],
        }

    monkeypatch.setattr(orchestrator_module, "run_chunk", fake_run_chunk)

    with sample_video.open("rb") as handle:
        upload_response = client.post(
            "/api/upload",
            files={"file": ("sample.mp4", handle, "video/mp4")},
        )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    task_id = payload["task_id"]

    start_response = client.post(
        f"/api/tasks/{task_id}/start",
        json={
            "llm_enabled": False,
            "video_enhancement_enabled": False,
            "text_model": "qwen2.5:7b",
            "video_model": "minicpm-v:8b",
        },
    )
    assert start_response.status_code == 200

    client.app.state.orchestrator.run_once()
    client.app.state.orchestrator.run_once()

    task_response = client.get(f"/api/tasks/{task_id}")
    assert task_response.status_code == 200
    assert task_response.json()["task"]["status"] == "completed"

    result_response = client.get(f"/api/tasks/{task_id}/result")
    assert result_response.status_code == 200
    result_payload = result_response.json()
    assert result_payload["total_segments"] >= 1

    artifact_response = client.get(f"/api/tasks/{task_id}/artifacts/result_json")
    assert artifact_response.status_code == 200

    chunk_files = client.app.state.repository.get_media_for_task(task_id, ArtifactKind.CHUNK_RESULT)
    assert len(chunk_files) >= 1
    chunk_artifact = client.get(
        f"/api/tasks/{task_id}/artifacts/chunk_result",
        params={"chunk_id": f"{task_id}-chunk-0000"},
    )
    assert chunk_artifact.status_code == 200

    debug_response = client.get(f"/api/tasks/{task_id}/debug")
    assert debug_response.status_code == 200
    debug_payload = debug_response.json()
    assert "segments" in debug_payload
    assert debug_payload["segments"][0]["track_count"] >= 1

    segment_debug = client.get(f"/api/tasks/{task_id}/debug/segments/0")
    assert segment_debug.status_code == 200
    assert "features" in segment_debug.json()

    rerun_response = client.post(
        f"/api/tasks/{task_id}/debug/segments/0/rerun",
        json={
            "mode": "images",
            "run_video": False,
            "run_text": False,
            "video_model": "minicpm-v:8b",
            "text_model": "qwen2.5:7b",
        },
    )
    assert rerun_response.status_code == 200
    rerun_payload = rerun_response.json()
    assert rerun_payload["segment"]["start_time"] >= 0
    assert "rerun" in rerun_payload


def test_upload_rejects_invalid_extension(client, tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("x", encoding="utf-8")
    with bad_file.open("rb") as handle:
        response = client.post("/api/upload", files={"file": ("bad.txt", handle, "text/plain")})
    assert response.status_code == 400


def test_llm_status_endpoint(client):
    response = client.get("/api/system/llm")
    assert response.status_code == 200
    payload = response.json()
    assert "text" in payload
    assert "video" in payload
    assert payload["text"]["provider"] in {"noop", "ollama_text"}
    assert payload["video"]["provider"] in {"noop", "ollama_vision"}


def test_models_status_endpoint(client):
    response = client.get("/api/system/models")
    assert response.status_code == 200
    payload = response.json()
    assert "available_text_models" in payload
    assert "available_video_models" in payload
