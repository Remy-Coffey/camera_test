from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import ArtifactKind, ChunkStatus, CleanupStatus, TaskStatus


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "track_id": self.track_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
            confidence=float(data["confidence"]),
            track_id=int(data["track_id"]) if data.get("track_id") is not None else None,
        )


@dataclass(slots=True)
class DetectionFrame:
    frame_index: int
    timestamp: float
    persons: list[BoundingBox] = field(default_factory=list)
    person_count: int = 0
    track_ids: list[int] = field(default_factory=list)
    scene_change_score: float = 0.0
    scene_changed: bool = False
    sampling_mode: str = "base"

    @property
    def has_person(self) -> bool:
        return bool(self.person_count or self.persons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "persons": [box.to_dict() for box in self.persons],
            "person_count": self.person_count or len(self.persons),
            "track_ids": self.track_ids or [box.track_id for box in self.persons if box.track_id is not None],
            "scene_change_score": self.scene_change_score,
            "scene_changed": self.scene_changed,
            "sampling_mode": self.sampling_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionFrame":
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp=float(data["timestamp"]),
            persons=[BoundingBox.from_dict(item) for item in data.get("persons", [])],
            person_count=int(data.get("person_count", len(data.get("persons", [])))),
            track_ids=[int(item) for item in data.get("track_ids", [])],
            scene_change_score=float(data.get("scene_change_score", 0.0)),
            scene_changed=bool(data.get("scene_changed", False)),
            sampling_mode=str(data.get("sampling_mode", "base")),
        )


@dataclass(slots=True)
class ActivitySegment:
    start_time: float
    end_time: float
    max_persons: int
    description: str
    thumbnail_timestamp: float
    rule_description: str
    features: dict[str, Any] = field(default_factory=dict)
    enhanced_description: str | None = None
    video_description: str | None = None
    video_labels: list[str] = field(default_factory=list)
    keyframe_timestamps: list[float] = field(default_factory=list)
    person_count_range: list[int] = field(default_factory=list)
    track_count: int = 0
    scene_change_score: float = 0.0
    action: str | None = None
    scene: str | None = None
    confidence: float | None = None
    debug_available: bool = False
    fallback_reason: str | None = None
    video_result_status: str = "rule_only"
    parse_mode: str | None = None
    raw_response_present: bool = False
    sampling_events: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "max_persons": self.max_persons,
            "description": self.description,
            "rule_description": self.rule_description,
            "enhanced_description": self.enhanced_description,
            "video_description": self.video_description,
            "video_labels": self.video_labels,
            "keyframe_timestamps": self.keyframe_timestamps,
            "person_count_range": self.person_count_range,
            "track_count": self.track_count,
            "scene_change_score": self.scene_change_score,
            "action": self.action,
            "scene": self.scene,
            "confidence": self.confidence,
            "debug_available": self.debug_available,
            "fallback_reason": self.fallback_reason,
            "video_result_status": self.video_result_status,
            "parse_mode": self.parse_mode,
            "raw_response_present": self.raw_response_present,
            "sampling_events": self.sampling_events,
            "debug": self.debug,
            "thumbnail_timestamp": self.thumbnail_timestamp,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActivitySegment":
        return cls(
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            max_persons=int(data["max_persons"]),
            description=str(data["description"]),
            thumbnail_timestamp=float(data["thumbnail_timestamp"]),
            rule_description=str(data.get("rule_description", data["description"])),
            features=dict(data.get("features", {})),
            enhanced_description=data.get("enhanced_description"),
            video_description=data.get("video_description"),
            video_labels=[str(item) for item in data.get("video_labels", [])],
            keyframe_timestamps=[float(item) for item in data.get("keyframe_timestamps", [])],
            person_count_range=[int(item) for item in data.get("person_count_range", [])],
            track_count=int(data.get("track_count", 0)),
            scene_change_score=float(data.get("scene_change_score", 0.0)),
            action=data.get("action"),
            scene=data.get("scene"),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            debug_available=bool(data.get("debug_available", False)),
            fallback_reason=data.get("fallback_reason"),
            video_result_status=str(data.get("video_result_status", "rule_only")),
            parse_mode=data.get("parse_mode"),
            raw_response_present=bool(data.get("raw_response_present", False)),
            sampling_events=[str(item) for item in data.get("sampling_events", [])],
            debug=dict(data.get("debug", {})),
        )


@dataclass(slots=True)
class AnalysisResult:
    video_duration: float
    total_frames_analyzed: int
    fps: float
    width: int
    height: int
    segments: list[ActivitySegment] = field(default_factory=list)
    generated_at: datetime | None = None
    result_source: str = "rule"
    text_model: str | None = None
    video_model: str | None = None
    text_enhancement_used: bool = False
    video_enhancement_used: bool = False
    debug_available: bool = False
    debug_summary: dict[str, Any] = field(default_factory=dict)
    sampling_profile: dict[str, Any] = field(default_factory=dict)

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_duration": self.video_duration,
            "total_frames_analyzed": self.total_frames_analyzed,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "segments": [segment.to_dict() for segment in self.segments],
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "result_source": self.result_source,
            "total_segments": self.total_segments,
            "text_model": self.text_model,
            "video_model": self.video_model,
            "text_enhancement_used": self.text_enhancement_used,
            "video_enhancement_used": self.video_enhancement_used,
            "debug_available": self.debug_available,
            "debug_summary": self.debug_summary,
            "sampling_profile": self.sampling_profile,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        generated_at = data.get("generated_at")
        return cls(
            video_duration=float(data["video_duration"]),
            total_frames_analyzed=int(data["total_frames_analyzed"]),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            segments=[ActivitySegment.from_dict(item) for item in data.get("segments", [])],
            generated_at=datetime.fromisoformat(generated_at) if generated_at else None,
            result_source=str(data.get("result_source", "rule")),
            text_model=data.get("text_model"),
            video_model=data.get("video_model"),
            text_enhancement_used=bool(data.get("text_enhancement_used", False)),
            video_enhancement_used=bool(data.get("video_enhancement_used", False)),
            debug_available=bool(data.get("debug_available", False)),
            debug_summary=dict(data.get("debug_summary", {})),
            sampling_profile=dict(data.get("sampling_profile", {})),
        )


@dataclass(slots=True)
class AnalysisTask:
    task_id: str
    status: TaskStatus
    progress: float
    stage: str
    video_file_id: str
    error: str | None = None
    llm_enabled: bool = False
    video_enhancement_enabled: bool = False
    text_model: str | None = None
    video_model: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    chunk_count: int = 0
    completed_chunks: int = 0
    processing_chunks: int = 0
    queued_chunks: int = 0
    recovery_stage: str | None = None
    recovery_reason: str | None = None
    last_recovered_at: datetime | None = None
    artifact_health: str = "unknown"


@dataclass(slots=True)
class AnalysisChunk:
    chunk_id: str
    task_id: str
    status: ChunkStatus
    start_time: float
    end_time: float
    overlap_seconds: float
    progress: float = 0.0
    attempt_count: int = 0
    error: str | None = None
    artifact_file_id: str | None = None
    artifact_path: str | None = None
    frame_count: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    result_payload: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class MediaFile:
    file_id: str
    task_id: str
    kind: ArtifactKind
    path: str
    original_name: str | None
    mime_type: str | None
    extension: str
    size_bytes: int
    sha256: str
    ttl_seconds: int | None
    expires_at: datetime | None
    deleted_at: datetime | None = None
    delete_status: CleanupStatus = CleanupStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
