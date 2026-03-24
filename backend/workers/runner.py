from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from detector import PersonDetector


@dataclass(slots=True)
class ChunkWorkItem:
    chunk_id: str
    task_id: str
    video_path: str
    model_path: str
    start_time: float
    end_time: float
    sample_interval: float
    stable_person_interval: float
    refined_sample_interval: float
    strong_refined_sample_interval: float


def run_chunk(item: ChunkWorkItem) -> dict[str, Any]:
    detector = PersonDetector(Path(item.model_path))
    chunk_result = detector.process_range(
        video_path=item.video_path,
        start_time=item.start_time,
        end_time=item.end_time,
        sample_interval=item.sample_interval,
        stable_person_interval=item.stable_person_interval,
        refined_sample_interval=item.refined_sample_interval,
        strong_refined_sample_interval=item.strong_refined_sample_interval,
    )
    return {
        "chunk_id": item.chunk_id,
        "task_id": item.task_id,
        "detections": [frame.to_dict() for frame in chunk_result.detections],
        "video_info": chunk_result.video_info,
        "frame_count": len(chunk_result.detections),
        "track_summary": chunk_result.track_summary,
        "sampling_profile": chunk_result.sampling_profile,
    }
