from pathlib import Path

from domain import ActivitySegment
from services.llm import DescriptionEnhancer, NoopProvider, OllamaVisionProvider, VideoEnhancer


def _segment() -> ActivitySegment:
    return ActivitySegment(
        start_time=0,
        end_time=1,
        max_persons=1,
        description="规则描述",
        rule_description="规则描述",
        thumbnail_timestamp=0.5,
        features={"movement": "moving"},
    )


def test_llm_noop_provider_degrades_gracefully():
    enhancer = DescriptionEnhancer(NoopProvider(), batch_size=3)
    assert enhancer.enhance([_segment()]) == [None]


def test_video_enhancer_noop_provider_degrades_gracefully(tmp_path: Path):
    enhancer = VideoEnhancer(NoopProvider(), batch_size=3)
    assert enhancer.analyze([_segment()], [[tmp_path / "fake.jpg"]]) == [None]


def test_vision_provider_accepts_weak_text_response(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake")
    provider = OllamaVisionProvider(
        base_url="http://127.0.0.1:11434",
        model="minicpm-v:8b",
        timeout_seconds=1,
        max_retries=1,
    )

    monkeypatch.setattr(provider, "health", lambda configured_model=None: {
        "reachable": True,
        "model_installed": True,
        "enabled": True,
        "provider": "ollama_vision",
        "base_url": "http://127.0.0.1:11434",
        "configured_model": configured_model or "minicpm-v:8b",
        "installed_models": ["minicpm-v:8b"],
    })
    monkeypatch.setattr(provider, "_post_json", lambda path, payload: {
        "ok": True,
        "data": {"message": {"content": "一个人站在室内镜头前，短暂停留并看向前方。"}},
        "status": "success",
        "error": None,
        "latency_ms": 12,
    })

    outputs = provider.analyze_segments_debug([_segment()], [[image_path]], batch_size=1)

    assert outputs[0] is not None
    assert outputs[0]["status"] == "weak_text_only"
    assert outputs[0]["video_result_status"] == "weak_success"
    assert outputs[0]["parse_mode"] == "weak_text"
    assert "一个人站在室内镜头前" in outputs[0]["output"]["description"]


def test_vision_provider_marks_low_quality_description(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake")
    provider = OllamaVisionProvider(
        base_url="http://127.0.0.1:11434",
        model="minicpm-v:8b",
        timeout_seconds=1,
        max_retries=1,
    )

    monkeypatch.setattr(provider, "health", lambda configured_model=None: {
        "reachable": True,
        "model_installed": True,
        "enabled": True,
        "provider": "ollama_vision",
        "base_url": "http://127.0.0.1:11434",
        "configured_model": configured_model or "minicpm-v:8b",
        "installed_models": ["minicpm-v:8b"],
    })
    monkeypatch.setattr(provider, "_post_json", lambda path, payload: {
        "ok": True,
        "data": {"message": {"content": "{\"description\":\"有人停留\",\"labels\":[]}"}},
        "status": "success",
        "error": None,
        "latency_ms": 12,
    })

    outputs = provider.analyze_segments_debug([_segment()], [[image_path]], batch_size=1)

    assert outputs[0]["status"] == "low_quality_description"
    assert outputs[0]["fallback_reason"] == "low_quality_description"
