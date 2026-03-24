from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timezone

from domain import ArtifactKind, CleanupStatus, TaskStatus
from repositories import SQLiteRepository


class CleanupService:
    def __init__(self, repository: SQLiteRepository, interval_seconds: int = 30):
        self.repository = repository
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="cleanup-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def run_once(self) -> None:
        for media in self.repository.list_expired_media():
            task = self.repository.get_task(media.task_id)
            if task and task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.EXPIRED):
                continue
            if media.kind in (ArtifactKind.THUMBNAIL, ArtifactKind.RESULT_JSON):
                continue
            try:
                if os.path.exists(media.path):
                    os.remove(media.path)
                deleted_at = datetime.now(timezone.utc)
                self.repository.mark_media_deleted(media.file_id, CleanupStatus.COMPLETED, deleted_at)
                self.repository.create_cleanup_job(str(uuid.uuid4()), media.file_id, CleanupStatus.COMPLETED)
            except OSError as exc:
                self.repository.mark_media_deleted(media.file_id, CleanupStatus.FAILED, None)
                self.repository.create_cleanup_job(
                    str(uuid.uuid4()), media.file_id, CleanupStatus.FAILED, str(exc)
                )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            time.sleep(self.interval_seconds)
