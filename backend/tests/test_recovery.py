from __future__ import annotations

import json
from pathlib import Path

from domain import AnalysisChunk, ArtifactKind, ChunkStatus, TaskStatus
from repositories import SQLiteRepository
from services.orchestrator import TaskOrchestrator
from services.video import probe_video


def _create_task_with_video(repository: SQLiteRepository, sample_video: Path):
    metadata = probe_video(sample_video)
    repository.create_task("task-1", "file-1", llm_enabled=False, initial_status=TaskStatus.PROCESSING)
    repository.create_media_file(
        file_id="file-1",
        task_id="task-1",
        kind=ArtifactKind.ORIGINAL_VIDEO,
        path=str(sample_video),
        original_name="sample.mp4",
        mime_type="video/mp4",
        extension=".mp4",
        size_bytes=metadata.size_bytes,
        sha256=metadata.sha256,
        ttl_seconds=3600,
        metadata={"duration_seconds": metadata.duration_seconds},
    )


def test_recovery_requeues_only_unfinished_chunks(temp_config, sample_video: Path):
    repository = SQLiteRepository(temp_config.database_path)
    _create_task_with_video(repository, sample_video)
    completed_artifact = temp_config.artifact_dir / "task-1" / "chunks" / "task-1-chunk-0000.json"
    completed_artifact.parent.mkdir(parents=True, exist_ok=True)
    completed_artifact.write_text(json.dumps({"chunk_id": "task-1-chunk-0000", "detections": []}), encoding="utf-8")
    repository.create_media_file(
        file_id="chunk-file-1",
        task_id="task-1",
        kind=ArtifactKind.CHUNK_RESULT,
        path=str(completed_artifact),
        original_name=completed_artifact.name,
        mime_type="application/json",
        extension=".json",
        size_bytes=completed_artifact.stat().st_size,
        sha256="x",
        ttl_seconds=3600,
        metadata={"chunk_id": "task-1-chunk-0000"},
    )
    repository.insert_chunks(
        [
            AnalysisChunk(
                chunk_id="task-1-chunk-0000",
                task_id="task-1",
                status=ChunkStatus.COMPLETED,
                start_time=0,
                end_time=1,
                overlap_seconds=0,
                artifact_file_id="chunk-file-1",
                artifact_path=str(completed_artifact),
            ),
            AnalysisChunk(
                chunk_id="task-1-chunk-0001",
                task_id="task-1",
                status=ChunkStatus.PROCESSING,
                start_time=1,
                end_time=2,
                overlap_seconds=0,
            ),
        ]
    )

    orchestrator = TaskOrchestrator(repository, temp_config)
    orchestrator.recover_incomplete_tasks()

    chunks = repository.list_chunks("task-1")
    task = repository.get_task("task-1")

    assert chunks[0].status == ChunkStatus.COMPLETED
    assert chunks[1].status == ChunkStatus.QUEUED
    assert task.status == TaskStatus.QUEUED
    assert task.recovery_reason == "resume_processing"


def test_recovery_merging_with_missing_chunk_artifact_falls_back_to_processing(temp_config, sample_video: Path):
    repository = SQLiteRepository(temp_config.database_path)
    _create_task_with_video(repository, sample_video)
    repository.update_task("task-1", status=TaskStatus.MERGING, stage="merging")
    repository.insert_chunks(
        [
            AnalysisChunk(
                chunk_id="task-1-chunk-0000",
                task_id="task-1",
                status=ChunkStatus.COMPLETED,
                start_time=0,
                end_time=1,
                overlap_seconds=0,
                artifact_path=str(temp_config.artifact_dir / "missing.json"),
            )
        ]
    )

    orchestrator = TaskOrchestrator(repository, temp_config)
    orchestrator.recover_incomplete_tasks()

    task = repository.get_task("task-1")
    chunk = repository.get_chunk("task-1-chunk-0000")

    assert task.status == TaskStatus.QUEUED
    assert task.recovery_reason == "missing_chunk_artifact_restart_processing"
    assert chunk.status == ChunkStatus.QUEUED
