from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from domain import (
    AnalysisChunk,
    AnalysisTask,
    ArtifactKind,
    ChunkStatus,
    CleanupStatus,
    MediaFile,
    TaskStatus,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def dt_to_str(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def str_to_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SQLiteRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage TEXT NOT NULL,
                    error TEXT,
                    video_file_id TEXT NOT NULL,
                    llm_enabled INTEGER NOT NULL DEFAULT 0,
                    video_enhancement_enabled INTEGER NOT NULL DEFAULT 0,
                    text_model TEXT,
                    video_model TEXT,
                    performance_profile TEXT NOT NULL DEFAULT 'balanced',
                    recovery_stage TEXT,
                    recovery_reason TEXT,
                    last_recovered_at TEXT,
                    artifact_health TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS analysis_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    overlap_seconds REAL NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    artifact_file_id TEXT,
                    artifact_path TEXT,
                    frame_count INTEGER NOT NULL DEFAULT 0,
                    summary_json TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES analysis_tasks(task_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    task_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    enhanced INTEGER NOT NULL DEFAULT 0,
                    result_source TEXT NOT NULL DEFAULT 'rule',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES analysis_tasks(task_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS media_files (
                    file_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    original_name TEXT,
                    mime_type TEXT,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    ttl_seconds INTEGER,
                    expires_at TEXT,
                    deleted_at TEXT,
                    delete_status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES analysis_tasks(task_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS cleanup_jobs (
                    job_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    scheduled_at TEXT NOT NULL,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(file_id) REFERENCES media_files(file_id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column(conn, "analysis_tasks", "recovery_stage", "TEXT")
            self._ensure_column(conn, "analysis_tasks", "recovery_reason", "TEXT")
            self._ensure_column(conn, "analysis_tasks", "last_recovered_at", "TEXT")
            self._ensure_column(conn, "analysis_tasks", "artifact_health", "TEXT NOT NULL DEFAULT 'unknown'")
            self._ensure_column(conn, "analysis_tasks", "video_enhancement_enabled", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "analysis_tasks", "text_model", "TEXT")
            self._ensure_column(conn, "analysis_tasks", "video_model", "TEXT")
            self._ensure_column(conn, "analysis_tasks", "performance_profile", "TEXT NOT NULL DEFAULT 'balanced'")
            self._ensure_column(conn, "analysis_chunks", "artifact_file_id", "TEXT")
            self._ensure_column(conn, "analysis_chunks", "artifact_path", "TEXT")
            self._ensure_column(conn, "analysis_chunks", "frame_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "analysis_chunks", "summary_json", "TEXT")
            self._ensure_column(conn, "analysis_results", "result_source", "TEXT NOT NULL DEFAULT 'rule'")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row["name"] == column for row in rows):
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    def create_task(
        self,
        task_id: str,
        video_file_id: str,
        llm_enabled: bool,
        video_enhancement_enabled: bool = False,
        text_model: str | None = None,
        video_model: str | None = None,
        performance_profile: str = "balanced",
        initial_status: TaskStatus = TaskStatus.QUEUED,
        stage: str = "uploaded",
    ) -> AnalysisTask:
        now = utcnow()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_tasks (
                    task_id, status, progress, stage, error, video_file_id,
                    llm_enabled, video_enhancement_enabled, text_model, video_model, performance_profile,
                    recovery_stage, recovery_reason, last_recovered_at,
                    artifact_health, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    initial_status.value,
                    0.0,
                    stage,
                    None,
                    video_file_id,
                    int(llm_enabled),
                    int(video_enhancement_enabled),
                    text_model,
                    video_model,
                    performance_profile,
                    None,
                    None,
                    None,
                    "unknown",
                    dt_to_str(now),
                    dt_to_str(now),
                ),
            )
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> AnalysisTask | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT t.*,
                       COUNT(c.chunk_id) AS chunk_count,
                       SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END) AS completed_chunks,
                       SUM(CASE WHEN c.status = 'processing' THEN 1 ELSE 0 END) AS processing_chunks,
                       SUM(CASE WHEN c.status = 'queued' THEN 1 ELSE 0 END) AS queued_chunks
                FROM analysis_tasks t
                LEFT JOIN analysis_chunks c ON c.task_id = t.task_id
                WHERE t.task_id = ?
                GROUP BY t.task_id
                """,
                (task_id,),
            ).fetchone()
        return self._row_to_task(row) if row else None

    def list_unfinished_tasks(self) -> list[AnalysisTask]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT t.*,
                       COUNT(c.chunk_id) AS chunk_count,
                       SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END) AS completed_chunks,
                       SUM(CASE WHEN c.status = 'processing' THEN 1 ELSE 0 END) AS processing_chunks,
                       SUM(CASE WHEN c.status = 'queued' THEN 1 ELSE 0 END) AS queued_chunks
                FROM analysis_tasks t
                LEFT JOIN analysis_chunks c ON c.task_id = t.task_id
                WHERE t.status IN ('queued', 'preparing', 'processing', 'merging', 'enhancing')
                GROUP BY t.task_id
                ORDER BY t.created_at ASC
                """
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = dt_to_str(utcnow())
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = [self._serialize_value(value) for value in fields.values()]
        with self.connection() as conn:
            conn.execute(
                f"UPDATE analysis_tasks SET {columns} WHERE task_id = ?",
                (*values, task_id),
            )

    def insert_chunks(self, chunks: list[AnalysisChunk]) -> None:
        if not chunks:
            return
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO analysis_chunks (
                    chunk_id, task_id, status, start_time, end_time, overlap_seconds,
                    progress, attempt_count, error, artifact_file_id, artifact_path,
                    frame_count, summary_json, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.task_id,
                        chunk.status.value,
                        chunk.start_time,
                        chunk.end_time,
                        chunk.overlap_seconds,
                        chunk.progress,
                        chunk.attempt_count,
                        chunk.error,
                        chunk.artifact_file_id,
                        chunk.artifact_path,
                        chunk.frame_count,
                        json.dumps(chunk.summary, ensure_ascii=False) if chunk.summary else None,
                        json.dumps(chunk.result_payload, ensure_ascii=False)
                        if chunk.result_payload
                        else None,
                        dt_to_str(chunk.created_at or utcnow()),
                        dt_to_str(chunk.updated_at or utcnow()),
                    )
                    for chunk in chunks
                ],
            )

    def list_chunks(self, task_id: str) -> list[AnalysisChunk]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM analysis_chunks WHERE task_id = ? ORDER BY start_time ASC",
                (task_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> AnalysisChunk | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def list_queued_chunks(self, limit: int) -> list[AnalysisChunk]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM analysis_chunks
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def update_chunk(self, chunk_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "result_payload" in fields:
            fields["result_json"] = json.dumps(fields.pop("result_payload"), ensure_ascii=False)
        if "summary" in fields:
            fields["summary_json"] = json.dumps(fields.pop("summary"), ensure_ascii=False)
        fields["updated_at"] = dt_to_str(utcnow())
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = [self._serialize_value(value) for value in fields.values()]
        with self.connection() as conn:
            conn.execute(
                f"UPDATE analysis_chunks SET {columns} WHERE chunk_id = ?",
                (*values, chunk_id),
            )

    def save_result(self, task_id: str, result_json: dict[str, Any], enhanced: bool, result_source: str) -> None:
        now = utcnow()
        payload = json.dumps(result_json, ensure_ascii=False)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_results (task_id, result_json, enhanced, result_source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    result_json = excluded.result_json,
                    enhanced = excluded.enhanced,
                    result_source = excluded.result_source,
                    updated_at = excluded.updated_at
                """,
                (task_id, payload, int(enhanced), result_source, dt_to_str(now), dt_to_str(now)),
            )

    def get_result(self, task_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT result_json FROM analysis_results WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def create_media_file(
        self,
        *,
        file_id: str,
        task_id: str,
        kind: ArtifactKind,
        path: str,
        original_name: str | None,
        mime_type: str | None,
        extension: str,
        size_bytes: int,
        sha256: str,
        ttl_seconds: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaFile:
        now = utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO media_files (
                    file_id, task_id, kind, path, original_name, mime_type, extension,
                    size_bytes, sha256, ttl_seconds, expires_at, deleted_at, delete_status,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    task_id,
                    kind.value,
                    path,
                    original_name,
                    mime_type,
                    extension,
                    size_bytes,
                    sha256,
                    ttl_seconds,
                    dt_to_str(expires_at),
                    None,
                    CleanupStatus.PENDING.value,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    dt_to_str(now),
                    dt_to_str(now),
                ),
            )
        return self.get_media_file(file_id)

    def get_media_file(self, file_id: str) -> MediaFile | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM media_files WHERE file_id = ?",
                (file_id,),
            ).fetchone()
        return self._row_to_media(row) if row else None

    def get_media_for_task(self, task_id: str, kind: ArtifactKind) -> list[MediaFile]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM media_files
                WHERE task_id = ? AND kind = ? AND deleted_at IS NULL
                ORDER BY created_at ASC
                """,
                (task_id, kind.value),
            ).fetchall()
        return [self._row_to_media(row) for row in rows]

    def mark_media_deleted(self, file_id: str, status: CleanupStatus, deleted_at: datetime | None) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE media_files
                SET delete_status = ?, deleted_at = ?, updated_at = ?
                WHERE file_id = ?
                """,
                (status.value, dt_to_str(deleted_at), dt_to_str(utcnow()), file_id),
            )

    def list_expired_media(self, limit: int = 20) -> list[MediaFile]:
        now = dt_to_str(utcnow())
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM media_files
                WHERE expires_at IS NOT NULL
                  AND deleted_at IS NULL
                  AND expires_at <= ?
                ORDER BY expires_at ASC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return [self._row_to_media(row) for row in rows]

    def create_cleanup_job(self, job_id: str, file_id: str, status: CleanupStatus, last_error: str | None = None) -> None:
        now = utcnow()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO cleanup_jobs (
                    job_id, file_id, status, attempt_count, last_error, scheduled_at,
                    completed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    file_id,
                    status.value,
                    1,
                    last_error,
                    dt_to_str(now),
                    dt_to_str(now) if status == CleanupStatus.COMPLETED else None,
                    dt_to_str(now),
                    dt_to_str(now),
                ),
            )

    def _row_to_task(self, row: sqlite3.Row) -> AnalysisTask:
        return AnalysisTask(
            task_id=row["task_id"],
            status=TaskStatus(row["status"]),
            progress=float(row["progress"]),
            stage=row["stage"],
            error=row["error"],
            video_file_id=row["video_file_id"],
            llm_enabled=bool(row["llm_enabled"]),
            video_enhancement_enabled=bool(row["video_enhancement_enabled"] if "video_enhancement_enabled" in row.keys() else 0),
            text_model=row["text_model"] if "text_model" in row.keys() else None,
            video_model=row["video_model"] if "video_model" in row.keys() else None,
            performance_profile=row["performance_profile"] if "performance_profile" in row.keys() and row["performance_profile"] else "balanced",
            created_at=str_to_dt(row["created_at"]),
            updated_at=str_to_dt(row["updated_at"]),
            started_at=str_to_dt(row["started_at"]),
            completed_at=str_to_dt(row["completed_at"]),
            chunk_count=int(row["chunk_count"] or 0),
            completed_chunks=int(row["completed_chunks"] or 0),
            processing_chunks=int(row["processing_chunks"] or 0),
            queued_chunks=int(row["queued_chunks"] or 0),
            recovery_stage=row["recovery_stage"],
            recovery_reason=row["recovery_reason"],
            last_recovered_at=str_to_dt(row["last_recovered_at"]),
            artifact_health=row["artifact_health"] or "unknown",
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> AnalysisChunk:
        return AnalysisChunk(
            chunk_id=row["chunk_id"],
            task_id=row["task_id"],
            status=ChunkStatus(row["status"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            overlap_seconds=float(row["overlap_seconds"]),
            progress=float(row["progress"]),
            attempt_count=int(row["attempt_count"]),
            error=row["error"],
            artifact_file_id=row["artifact_file_id"],
            artifact_path=row["artifact_path"],
            frame_count=int(row["frame_count"] or 0),
            summary=json.loads(row["summary_json"]) if row["summary_json"] else {},
            result_payload=json.loads(row["result_json"]) if row["result_json"] else None,
            created_at=str_to_dt(row["created_at"]),
            updated_at=str_to_dt(row["updated_at"]),
        )

    def _row_to_media(self, row: sqlite3.Row) -> MediaFile:
        return MediaFile(
            file_id=row["file_id"],
            task_id=row["task_id"],
            kind=ArtifactKind(row["kind"]),
            path=row["path"],
            original_name=row["original_name"],
            mime_type=row["mime_type"],
            extension=row["extension"],
            size_bytes=int(row["size_bytes"]),
            sha256=row["sha256"],
            ttl_seconds=int(row["ttl_seconds"]) if row["ttl_seconds"] is not None else None,
            expires_at=str_to_dt(row["expires_at"]),
            deleted_at=str_to_dt(row["deleted_at"]),
            delete_status=CleanupStatus(row["delete_status"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, (TaskStatus, ChunkStatus, ArtifactKind, CleanupStatus)):
            return value.value
        if isinstance(value, datetime):
            return dt_to_str(value)
        return value
