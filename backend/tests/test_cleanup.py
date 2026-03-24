from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from domain import ArtifactKind
from repositories import SQLiteRepository
from repositories.sqlite import utcnow
from services.cleanup import CleanupService


def test_cleanup_removes_expired_raw_video(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "db.sqlite3")
    repository.create_task(task_id="task-1", video_file_id="file-1", llm_enabled=False)
    target = tmp_path / "video.mp4"
    target.write_bytes(b"video")
    media = repository.create_media_file(
        file_id="file-1",
        task_id="task-1",
        kind=ArtifactKind.ORIGINAL_VIDEO,
        path=str(target),
        original_name="video.mp4",
        mime_type="video/mp4",
        extension=".mp4",
        size_bytes=target.stat().st_size,
        sha256="sha",
        ttl_seconds=1,
        metadata={},
    )
    with repository.connection() as conn:
        conn.execute(
            "UPDATE media_files SET expires_at = ? WHERE file_id = ?",
            ((utcnow() - timedelta(seconds=5)).isoformat(), media.file_id),
        )
        conn.execute(
            "UPDATE analysis_tasks SET status = 'completed' WHERE task_id = ?",
            ("task-1",),
        )

    service = CleanupService(repository, interval_seconds=1)
    service.run_once()

    assert not target.exists()
    assert repository.get_media_file("file-1").deleted_at is not None
