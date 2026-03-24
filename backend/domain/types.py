from __future__ import annotations

from enum import Enum


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    PROCESSING = "processing"
    MERGING = "merging"
    ENHANCING = "enhancing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


class ChunkStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ArtifactKind(str, Enum):
    ORIGINAL_VIDEO = "original_video"
    THUMBNAIL = "thumbnail"
    RESULT_JSON = "result_json"
    CHUNK_RESULT = "chunk_result"


class CleanupStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
