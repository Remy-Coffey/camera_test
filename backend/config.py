from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


TEXT_MODEL_WHITELIST = ("qwen2.5:7b", "mistral:7b-instruct-q4_K_M")
VIDEO_MODEL_WHITELIST = ("minicpm-v:8b", "moondream:1.8b", "llava:7b")
PERFORMANCE_PROFILES = {
    "fast": {
        "max_keyframes": 3,
        "keyframe_max_size": 512,
        "text_skip_min_chars": 40,
        "prefer_clip_rerun": False,
    },
    "balanced": {
        "max_keyframes": 4,
        "keyframe_max_size": 640,
        "text_skip_min_chars": 48,
        "prefer_clip_rerun": False,
    },
    "quality": {
        "max_keyframes": 5,
        "keyframe_max_size": 768,
        "text_skip_min_chars": 64,
        "prefer_clip_rerun": False,
    },
}


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    upload_dir: Path
    artifact_dir: Path
    frontend_dir: Path
    database_path: Path
    detector_model_path: Path
    sample_interval_seconds: float = 1.0
    stable_person_interval_seconds: float = 0.75
    refined_sample_interval_seconds: float = 0.25
    strong_refined_sample_interval_seconds: float = 0.1
    chunk_duration_seconds: int = 120
    chunk_overlap_seconds: int = 2
    max_workers: int = 2
    upload_max_bytes: int = 1024 * 1024 * 1024
    raw_video_ttl_seconds: int = 60 * 60 * 24 * 7
    chunk_artifact_ttl_seconds: int = 60 * 60 * 24
    llm_enabled: bool = False
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_timeout_seconds: int = 90
    llm_max_retries: int = 1
    text_batch_size: int = 5
    video_llm_enabled: bool = True
    video_llm_model: str = "minicpm-v:8b"
    video_batch_size: int = 3
    keyframes_per_segment: int = 4
    keyframe_max_size: int = 896
    worker_mode: str = "thread"
    default_performance_profile: str = "balanced"

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "AppConfig":
        root = base_dir or Path(__file__).resolve().parent.parent
        upload_dir = root / "uploads"
        artifact_dir = root / "artifacts"
        frontend_dir = root / "frontend"
        database_path = root / "camera_test.sqlite3"
        detector_model_path = Path(
            os.getenv("CAMERA_TEST_MODEL_PATH", str(root / "backend" / "yolov8n.pt"))
        )
        default_worker_mode = "thread" if platform.system() == "Windows" else "process"
        return cls(
            base_dir=root,
            upload_dir=upload_dir,
            artifact_dir=artifact_dir,
            frontend_dir=frontend_dir,
            database_path=Path(os.getenv("CAMERA_TEST_DB_PATH", str(database_path))),
            detector_model_path=detector_model_path,
            sample_interval_seconds=float(os.getenv("CAMERA_TEST_SAMPLE_INTERVAL", "1.0")),
            stable_person_interval_seconds=float(os.getenv("CAMERA_TEST_STABLE_PERSON_INTERVAL", "0.75")),
            refined_sample_interval_seconds=float(os.getenv("CAMERA_TEST_REFINED_SAMPLE_INTERVAL", "0.25")),
            strong_refined_sample_interval_seconds=float(os.getenv("CAMERA_TEST_STRONG_REFINED_SAMPLE_INTERVAL", "0.1")),
            chunk_duration_seconds=int(os.getenv("CAMERA_TEST_CHUNK_SECONDS", "120")),
            chunk_overlap_seconds=int(os.getenv("CAMERA_TEST_CHUNK_OVERLAP_SECONDS", "2")),
            max_workers=max(1, int(os.getenv("CAMERA_TEST_MAX_WORKERS", "2"))),
            upload_max_bytes=int(
                os.getenv("CAMERA_TEST_UPLOAD_MAX_BYTES", str(1024 * 1024 * 1024))
            ),
            raw_video_ttl_seconds=int(
                os.getenv("CAMERA_TEST_RAW_VIDEO_TTL_SECONDS", str(60 * 60 * 24 * 7))
            ),
            chunk_artifact_ttl_seconds=int(
                os.getenv("CAMERA_TEST_CHUNK_ARTIFACT_TTL_SECONDS", str(60 * 60 * 24))
            ),
            llm_enabled=os.getenv("CAMERA_TEST_LLM_ENABLED", "0") == "1",
            llm_model=os.getenv("CAMERA_TEST_LLM_MODEL", "qwen2.5:7b"),
            llm_base_url=os.getenv("CAMERA_TEST_LLM_BASE_URL", "http://127.0.0.1:11434"),
            llm_timeout_seconds=int(os.getenv("CAMERA_TEST_LLM_TIMEOUT_SECONDS", "90")),
            llm_max_retries=int(os.getenv("CAMERA_TEST_LLM_MAX_RETRIES", "1")),
            text_batch_size=max(1, int(os.getenv("CAMERA_TEST_TEXT_BATCH_SIZE", "5"))),
            video_llm_enabled=os.getenv("CAMERA_TEST_VIDEO_LLM_ENABLED", "1") == "1",
            video_llm_model=os.getenv("CAMERA_TEST_VIDEO_LLM_MODEL", "minicpm-v:8b"),
            video_batch_size=max(1, int(os.getenv("CAMERA_TEST_VIDEO_BATCH_SIZE", "3"))),
            keyframes_per_segment=max(2, int(os.getenv("CAMERA_TEST_KEYFRAMES_PER_SEGMENT", "4"))),
            keyframe_max_size=max(256, int(os.getenv("CAMERA_TEST_KEYFRAME_MAX_SIZE", "896"))),
            worker_mode=os.getenv("CAMERA_TEST_WORKER_MODE", default_worker_mode),
            default_performance_profile=os.getenv("CAMERA_TEST_PERFORMANCE_PROFILE", "balanced"),
        )

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available_text_models(self) -> tuple[str, ...]:
        return TEXT_MODEL_WHITELIST

    @property
    def available_video_models(self) -> tuple[str, ...]:
        return VIDEO_MODEL_WHITELIST

    @property
    def available_performance_profiles(self) -> tuple[str, ...]:
        return tuple(PERFORMANCE_PROFILES.keys())

    def resolve_performance_profile(self, value: str | None) -> str:
        candidate = (value or self.default_performance_profile or "balanced").strip().lower()
        if candidate not in PERFORMANCE_PROFILES:
            return "balanced"
        return candidate

    def performance_profile_settings(self, value: str | None) -> dict[str, int | bool]:
        profile = self.resolve_performance_profile(value)
        return {"name": profile, **PERFORMANCE_PROFILES[profile]}
