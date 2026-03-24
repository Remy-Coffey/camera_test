from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from domain import ArtifactKind, TaskStatus
from repositories import SQLiteRepository
from repositories.sqlite import utcnow
from services.cleanup import CleanupService


def test_cleanup_keeps_expired_chunk_artifact_for_unfinished_task(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "db.sqlite3")
    repository.create_task(task_id="task-1", video_file_id="file-1", llm_enabled=False, initial_status=TaskStatus.PROCESSING)
    target = tmp_path / "chunk.json"
    target.write_text("{}", encoding="utf-8")
    media = repository.create_media_file(
        file_id="chunk-file-1",
        task_id="task-1",
        kind=ArtifactKind.CHUNK_RESULT,
        path=str(target),
        original_name="chunk.json",
        mime_type="application/json",
        extension=".json",
        size_bytes=target.stat().st_size,
        sha256="sha",
        ttl_seconds=1,
        metadata={"chunk_id": "task-1-chunk-0000"},
    )
    with repository.connection() as conn:
        conn.execute(
            "UPDATE media_files SET expires_at = ? WHERE file_id = ?",
            ((utcnow() - timedelta(seconds=5)).isoformat(), media.file_id),
        )

    service = CleanupService(repository, interval_seconds=1)
    service.run_once()

    assert target.exists()
    assert repository.get_media_file(media.file_id).deleted_at is None
