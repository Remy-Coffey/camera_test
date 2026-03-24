from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import AppConfig
from domain import AnalysisChunk, AnalysisResult, ArtifactKind, ChunkStatus, DetectionFrame, TaskStatus
from repositories import SQLiteRepository
from repositories.sqlite import utcnow
from services.analysis import (
    apply_enhanced_descriptions,
    apply_video_insights,
    build_analysis_result,
    mark_text_fallback,
    mark_video_fallback,
)
from services.chunking import ChunkPlanner
from services.llm import DescriptionEnhancer, NoopProvider, OllamaTextProvider, OllamaVisionProvider, VideoEnhancer
from services.video import export_segment_clip, extract_segment_keyframes, probe_video, save_thumbnail
from workers import ChunkWorkItem, run_chunk


class InlineExecutor:
    def submit(self, fn: Callable[..., dict], *args, **kwargs) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True) -> None:
        return None


@dataclass(slots=True)
class StartTaskResult:
    task_id: str
    chunk_count: int


class TaskOrchestrator:
    def __init__(self, repository: SQLiteRepository, config: AppConfig):
        self.repository = repository
        self.config = config
        self.chunk_planner = ChunkPlanner(
            chunk_duration_seconds=config.chunk_duration_seconds,
            overlap_seconds=config.chunk_overlap_seconds,
        )
        if config.worker_mode == "inline":
            self.executor: concurrent.futures.Executor = InlineExecutor()
        elif config.worker_mode == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)
        else:
            try:
                self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=config.max_workers)
            except (OSError, PermissionError):
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)

        text_provider = (
            OllamaTextProvider(
                base_url=config.llm_base_url,
                model=config.llm_model,
                timeout_seconds=config.llm_timeout_seconds,
                max_retries=config.llm_max_retries,
            )
            if config.llm_enabled
            else NoopProvider()
        )
        vision_provider = (
            OllamaVisionProvider(
                base_url=config.llm_base_url,
                model=config.video_llm_model,
                timeout_seconds=config.llm_timeout_seconds,
                max_retries=config.llm_max_retries,
            )
            if config.video_llm_enabled
            else NoopProvider()
        )
        self.description_enhancer = DescriptionEnhancer(text_provider, config.text_batch_size)
        self.video_enhancer = VideoEnhancer(vision_provider, config.video_batch_size)
        self._futures: dict[concurrent.futures.Future, str] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.recover_incomplete_tasks()
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="task-orchestrator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.executor.shutdown(wait=False)

    def llm_status(self) -> dict[str, dict]:
        return {
            "text": self.description_enhancer.health(self.config.llm_model),
            "video": self.video_enhancer.health(self.config.video_llm_model),
        }

    def models_status(self) -> dict[str, object]:
        return {
            "default_text_model": self.config.llm_model,
            "default_video_model": self.config.video_llm_model,
            "available_text_models": list(self.config.available_text_models),
            "available_video_models": list(self.config.available_video_models),
            "text": self.description_enhancer.health(self.config.llm_model),
            "video": self.video_enhancer.health(self.config.video_llm_model),
        }

    def recover_incomplete_tasks(self) -> None:
        for task in self.repository.list_unfinished_tasks():
            self._recover_task(task.task_id)

    def prepare_task(
        self,
        task_id: str,
        *,
        llm_enabled: bool | None = None,
        video_enhancement_enabled: bool | None = None,
        text_model: str | None = None,
        video_model: str | None = None,
    ) -> StartTaskResult:
        task = self.repository.get_task(task_id)
        if not task:
            raise ValueError("task not found")
        media_files = self.repository.get_media_for_task(task_id, ArtifactKind.ORIGINAL_VIDEO)
        if not media_files:
            raise ValueError("original video not found")
        if text_model and text_model not in self.config.available_text_models:
            raise ValueError("unsupported text model")
        if video_model and video_model not in self.config.available_video_models:
            raise ValueError("unsupported video model")

        metadata = probe_video(Path(media_files[0].path))
        self.repository.update_task(
            task_id,
            status=TaskStatus.PREPARING,
            stage="planning_chunks",
            progress=max(task.progress, 5.0),
            started_at=task.started_at or utcnow(),
            artifact_health="checking",
            llm_enabled=self.config.llm_enabled if llm_enabled is None else llm_enabled,
            video_enhancement_enabled=self.config.video_llm_enabled if video_enhancement_enabled is None else video_enhancement_enabled,
            text_model=text_model or self.config.llm_model,
            video_model=video_model or self.config.video_llm_model,
            error=None,
        )

        chunks = self.repository.list_chunks(task_id)
        if chunks:
            return StartTaskResult(task_id=task_id, chunk_count=len(chunks))

        planned_chunks = self.chunk_planner.plan(metadata.duration_seconds)
        chunk_rows = [
            AnalysisChunk(
                chunk_id=f"{task_id}-chunk-{spec.index:04d}",
                task_id=task_id,
                status=ChunkStatus.QUEUED,
                start_time=spec.start_time,
                end_time=spec.end_time,
                overlap_seconds=spec.overlap_seconds,
            )
            for spec in planned_chunks
        ]
        self.repository.insert_chunks(chunk_rows)
        self.repository.update_task(task_id, status=TaskStatus.QUEUED, stage="queued", progress=10.0, artifact_health="healthy")
        return StartTaskResult(task_id=task_id, chunk_count=len(chunk_rows))

    def cancel_task(self, task_id: str) -> None:
        task = self.repository.get_task(task_id)
        if not task:
            raise ValueError("task not found")
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.EXPIRED):
            return
        for chunk in self.repository.list_chunks(task_id):
            if chunk.status in (ChunkStatus.QUEUED, ChunkStatus.PROCESSING):
                self.repository.update_chunk(chunk.chunk_id, status=ChunkStatus.CANCELLED, error="cancelled by user")
        self.repository.update_task(task_id, status=TaskStatus.CANCELLED, stage="cancelled", error="cancelled by user")

    def run_once(self) -> None:
        self._collect_finished_work()
        self._resume_stage_tasks()
        available_slots = self.config.max_workers - len(self._futures)
        if available_slots <= 0:
            return
        for chunk in self.repository.list_queued_chunks(limit=available_slots):
            self._submit_chunk(chunk)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            time.sleep(0.5)

    def _recover_task(self, task_id: str) -> None:
        task = self.repository.get_task(task_id)
        if not task:
            return
        original_media = self.repository.get_media_for_task(task_id, ArtifactKind.ORIGINAL_VIDEO)
        if not original_media or not Path(original_media[0].path).exists():
            self.repository.update_task(
                task_id,
                status=TaskStatus.FAILED,
                stage="failed",
                error="original video missing",
                recovery_stage=task.stage,
                recovery_reason="original_video_missing",
                last_recovered_at=utcnow(),
                artifact_health="broken",
            )
            return
        chunks = self.repository.list_chunks(task_id)
        missing_chunk_artifacts = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.status == ChunkStatus.COMPLETED and (not chunk.artifact_path or not Path(chunk.artifact_path).exists())
        ]
        if task.status in (TaskStatus.QUEUED, TaskStatus.PREPARING):
            self.repository.update_task(
                task_id,
                status=TaskStatus.QUEUED,
                stage="recovered_queue",
                error=None,
                recovery_stage=task.stage,
                recovery_reason="requeue_pending_task",
                last_recovered_at=utcnow(),
                artifact_health="healthy",
            )
            return
        if task.status == TaskStatus.PROCESSING:
            for chunk in chunks:
                if chunk.status == ChunkStatus.PROCESSING or chunk.chunk_id in missing_chunk_artifacts:
                    self.repository.update_chunk(chunk.chunk_id, status=ChunkStatus.QUEUED, progress=0.0)
            self.repository.update_task(
                task_id,
                status=TaskStatus.QUEUED,
                stage="recovered_processing",
                error=None,
                recovery_stage=task.stage,
                recovery_reason="resume_processing",
                last_recovered_at=utcnow(),
                artifact_health="degraded" if missing_chunk_artifacts else "healthy",
            )
            return
        if task.status == TaskStatus.MERGING:
            if missing_chunk_artifacts:
                for chunk in chunks:
                    if chunk.chunk_id in missing_chunk_artifacts:
                        self.repository.update_chunk(chunk.chunk_id, status=ChunkStatus.QUEUED, progress=0.0)
                self.repository.update_task(
                    task_id,
                    status=TaskStatus.QUEUED,
                    stage="recovered_processing",
                    error=None,
                    recovery_stage=task.stage,
                    recovery_reason="missing_chunk_artifact_restart_processing",
                    last_recovered_at=utcnow(),
                    artifact_health="degraded",
                )
            else:
                self.repository.update_task(
                    task_id,
                    status=TaskStatus.MERGING,
                    stage="recovered_merging",
                    error=None,
                    recovery_stage=task.stage,
                    recovery_reason="resume_merging",
                    last_recovered_at=utcnow(),
                    artifact_health="healthy",
                )
            return
        if task.status == TaskStatus.ENHANCING:
            if not self.repository.get_result(task_id):
                self.repository.update_task(
                    task_id,
                    status=TaskStatus.MERGING,
                    stage="recovered_merging",
                    error=None,
                    recovery_stage=task.stage,
                    recovery_reason="result_missing_before_enhancing",
                    last_recovered_at=utcnow(),
                    artifact_health="degraded",
                )
            else:
                self.repository.update_task(
                    task_id,
                    status=TaskStatus.ENHANCING,
                    stage="recovered_enhancing",
                    error=None,
                    recovery_stage=task.stage,
                    recovery_reason="resume_enhancing",
                    last_recovered_at=utcnow(),
                    artifact_health="healthy",
                )

    def _resume_stage_tasks(self) -> None:
        for task in self.repository.list_unfinished_tasks():
            if task.status == TaskStatus.MERGING:
                chunks = self.repository.list_chunks(task.task_id)
                if chunks and all(chunk.status == ChunkStatus.COMPLETED for chunk in chunks):
                    self._finalize_task(task.task_id, chunks)
            elif task.status == TaskStatus.ENHANCING:
                self._resume_enhancement(task.task_id)

    def _submit_chunk(self, chunk: AnalysisChunk) -> None:
        task = self.repository.get_task(chunk.task_id)
        if not task or task.status == TaskStatus.CANCELLED:
            return
        media_files = self.repository.get_media_for_task(chunk.task_id, ArtifactKind.ORIGINAL_VIDEO)
        if not media_files:
            self.repository.update_task(chunk.task_id, status=TaskStatus.FAILED, stage="failed", error="original video missing")
            return
        work_item = ChunkWorkItem(
            chunk_id=chunk.chunk_id,
            task_id=chunk.task_id,
            video_path=media_files[0].path,
            model_path=str(self.config.detector_model_path),
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            sample_interval=self.config.sample_interval_seconds,
            stable_person_interval=self.config.stable_person_interval_seconds,
            refined_sample_interval=self.config.refined_sample_interval_seconds,
            strong_refined_sample_interval=self.config.strong_refined_sample_interval_seconds,
        )
        self.repository.update_chunk(
            chunk.chunk_id,
            status=ChunkStatus.PROCESSING,
            progress=5.0,
            attempt_count=chunk.attempt_count + 1,
            error=None,
        )
        self.repository.update_task(chunk.task_id, status=TaskStatus.PROCESSING, stage="chunk_processing", error=None)
        future = self.executor.submit(run_chunk, work_item)
        with self._lock:
            self._futures[future] = chunk.chunk_id

    def _collect_finished_work(self) -> None:
        with self._lock:
            active_futures = list(self._futures.keys())
        completed_futures = [future for future in active_futures if future.done()]
        if not completed_futures:
            return
        for future in completed_futures:
            chunk_id = self._futures.get(future)
            if not chunk_id:
                continue
            chunk = self.repository.get_chunk(chunk_id)
            if not chunk:
                continue
            try:
                payload = future.result()
                artifact = self._persist_chunk_artifact(chunk, payload)
                self.repository.update_chunk(
                    chunk_id,
                    status=ChunkStatus.COMPLETED,
                    progress=100.0,
                    artifact_file_id=artifact.file_id,
                    artifact_path=artifact.path,
                    frame_count=int(payload.get("frame_count", 0)),
                    summary={
                        "frame_count": int(payload.get("frame_count", 0)),
                        "track_count": len(payload.get("track_summary", [])),
                        "sampling_profile": payload.get("sampling_profile", {}),
                    },
                    result_payload=payload,
                    error=None,
                )
            except Exception as exc:  # pragma: no cover
                self.repository.update_chunk(chunk_id, status=ChunkStatus.FAILED, error=str(exc), progress=0.0)
                self.repository.update_task(chunk.task_id, status=TaskStatus.FAILED, stage="failed", error=str(exc))
                continue
            self._update_task_progress(chunk.task_id)
            self._maybe_finalize_task(chunk.task_id)
        with self._lock:
            for future in completed_futures:
                self._futures.pop(future, None)

    def _persist_chunk_artifact(self, chunk: AnalysisChunk, payload: dict):
        artifact_path = self.config.artifact_dir / chunk.task_id / "chunks" / f"{chunk.chunk_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for media in self.repository.get_media_for_task(chunk.task_id, ArtifactKind.CHUNK_RESULT):
            if media.path == str(artifact_path):
                return media
        return self.repository.create_media_file(
            file_id=str(uuid.uuid4()),
            task_id=chunk.task_id,
            kind=ArtifactKind.CHUNK_RESULT,
            path=str(artifact_path),
            original_name=artifact_path.name,
            mime_type="application/json",
            extension=".json",
            size_bytes=artifact_path.stat().st_size,
            sha256=f"chunk-{chunk.chunk_id}",
            ttl_seconds=self.config.chunk_artifact_ttl_seconds,
            metadata={"chunk_id": chunk.chunk_id},
        )

    def _update_task_progress(self, task_id: str) -> None:
        task = self.repository.get_task(task_id)
        if not task or task.chunk_count <= 0:
            return
        weighted_completed = task.completed_chunks + (task.processing_chunks * 0.35)
        progress = 10.0 + (weighted_completed / task.chunk_count) * 70.0
        self.repository.update_task(task_id, progress=round(progress, 2), stage="chunk_processing")

    def _maybe_finalize_task(self, task_id: str) -> None:
        task = self.repository.get_task(task_id)
        if not task or task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED) or task.chunk_count == 0:
            return
        if task.completed_chunks != task.chunk_count:
            return
        chunks = self.repository.list_chunks(task_id)
        if any(chunk.status != ChunkStatus.COMPLETED for chunk in chunks):
            return
        self._finalize_task(task_id, chunks)

    def _finalize_task(self, task_id: str, chunks: list[AnalysisChunk]) -> None:
        self.repository.update_task(task_id, status=TaskStatus.MERGING, stage="merging", progress=85.0)
        merged = self._merge_task_chunks(task_id, chunks)
        self.repository.save_result(task_id, merged.to_dict(), enhanced=False, result_source="rule")
        task = self.repository.get_task(task_id)
        if task and ((task.video_enhancement_enabled and self._video_model_available(task)) or (task.llm_enabled and self._text_model_available(task))):
            self.repository.update_task(task_id, status=TaskStatus.ENHANCING, stage="enhancing", progress=90.0)
            self._resume_enhancement(task_id)
        else:
            self._persist_result_artifacts(task_id, merged)
            self.repository.update_task(task_id, status=TaskStatus.COMPLETED, stage="completed", progress=100.0, error=None, completed_at=utcnow())

    def _resume_enhancement(self, task_id: str) -> None:
        raw_result = self.repository.get_result(task_id)
        if not raw_result:
            chunks = self.repository.list_chunks(task_id)
            self._finalize_task(task_id, chunks)
            return
        result = AnalysisResult.from_dict(raw_result)
        task = self.repository.get_task(task_id)
        if not task:
            return

        if task.video_enhancement_enabled:
            if self._video_model_available(task):
                self.repository.update_task(task_id, status=TaskStatus.ENHANCING, stage="video_enhancing", progress=max(task.progress, 90.0))
                result = self._apply_video_enhancement(task, result)
            else:
                result = mark_video_fallback(result, "video model unavailable", task.video_model or self.config.video_llm_model)
            self.repository.save_result(task_id, result.to_dict(), enhanced=result.video_enhancement_used, result_source=result.result_source)

        if task.llm_enabled:
            if self._text_model_available(task):
                self.repository.update_task(task_id, status=TaskStatus.ENHANCING, stage="enhancing", progress=max(task.progress, 95.0))
                result = self._apply_text_enhancement(task, result)
            else:
                result = mark_text_fallback(result, "text model unavailable", task.text_model or self.config.llm_model)
            self.repository.save_result(
                task_id,
                result.to_dict(),
                enhanced=result.text_enhancement_used or result.video_enhancement_used,
                result_source=result.result_source,
            )

        self._persist_result_artifacts(task_id, result)
        self.repository.update_task(task_id, status=TaskStatus.COMPLETED, stage="completed", progress=100.0, error=None, completed_at=utcnow())

    def _video_model_available(self, task) -> bool:
        configured = task.video_model or self.config.video_llm_model
        status = self.video_enhancer.health(configured)
        return bool(status["reachable"] and status["model_installed"])

    def _text_model_available(self, task) -> bool:
        configured = task.text_model or self.config.llm_model
        status = self.description_enhancer.health(configured)
        return bool(status["reachable"] and status["model_installed"])

    def _select_keyframe_timestamps(self, segment, detections: list[DetectionFrame]) -> list[float]:
        segment_frames = [frame for frame in detections if segment.start_time <= frame.timestamp <= segment.end_time and frame.has_person]
        if not segment_frames:
            midpoint = segment.start_time + max(segment.end_time - segment.start_time, 0.1) / 2.0
            return [round(midpoint, 3)]

        selected: list[DetectionFrame] = [segment_frames[0]]
        midpoint = segment_frames[len(segment_frames) // 2]
        if midpoint.timestamp not in {frame.timestamp for frame in selected}:
            selected.append(midpoint)

        for frame in segment_frames:
            if frame.scene_changed or frame.sampling_mode == "refined":
                if frame.timestamp not in {item.timestamp for item in selected}:
                    selected.append(frame)

        for prev, curr in zip(segment_frames, segment_frames[1:]):
            if prev.person_count != curr.person_count or (prev.track_ids and curr.track_ids and prev.track_ids[0] != curr.track_ids[0]):
                if curr.timestamp not in {item.timestamp for item in selected}:
                    selected.append(curr)

        if segment_frames[-1].timestamp not in {frame.timestamp for frame in selected}:
            selected.append(segment_frames[-1])

        selected = sorted(selected, key=lambda item: item.timestamp)
        unique: list[float] = []
        for frame in selected:
            if frame.timestamp not in unique:
                unique.append(frame.timestamp)
        return unique[: max(self.config.keyframes_per_segment + 1, 5)]

    def _apply_video_enhancement(self, task, result: AnalysisResult) -> AnalysisResult:
        media = self.repository.get_media_for_task(task.task_id, ArtifactKind.ORIGINAL_VIDEO)
        if not media or not result.segments:
            return mark_video_fallback(result, "missing video or segments", task.video_model or self.config.video_llm_model)

        video_path = Path(media[0].path)
        chunks = self.repository.list_chunks(task.task_id)
        all_frames: dict[float, DetectionFrame] = {}
        for chunk in chunks:
            payload = self._load_chunk_payload(chunk)
            for item in payload.get("detections", []):
                frame = DetectionFrame.from_dict(item)
                all_frames[frame.timestamp] = frame
        ordered_frames = [all_frames[key] for key in sorted(all_frames.keys())]

        segment_images: list[list[Path]] = []
        keyframe_timestamps: list[list[float]] = []
        total = max(1, len(result.segments))
        for index, segment in enumerate(result.segments, start=1):
            timestamps = self._select_keyframe_timestamps(segment, ordered_frames)
            frame_dir = self.config.artifact_dir / task.task_id / "keyframes" / f"{index - 1:04d}"
            keyframes = extract_segment_keyframes(video_path, timestamps, frame_dir, self.config.keyframe_max_size)
            segment_images.append([path for path, _ in keyframes])
            keyframe_timestamps.append([timestamp for _, timestamp in keyframes])
            progress = 90.0 + (index / total) * 3.0
            self.repository.update_task(task.task_id, status=TaskStatus.ENHANCING, stage="video_enhancing", progress=round(progress, 2))

        insights = self.video_enhancer.analyze(result.segments, segment_images, model=task.video_model or self.config.video_llm_model)
        enriched_insights: list[dict | None] = []
        for index, (segment, insight, images, timestamps) in enumerate(zip(result.segments, insights, segment_images, keyframe_timestamps)):
            payload = insight or {
                "output": None,
                "fallback_reason": "empty_video_description",
                "video_result_status": "fallback",
                "parse_mode": None,
                "raw_response_present": False,
                "debug": {},
            }
            debug_payload = dict(payload.get("debug", {}))
            debug_payload["rule_summary"] = {
                "rule_description": segment.rule_description,
                "max_persons": segment.max_persons,
                "person_count_range": segment.person_count_range,
                "track_count": segment.track_count,
                "scene_change_score": segment.scene_change_score,
                "sampling_events": segment.sampling_events,
                "direction": segment.features.get("direction"),
            }
            debug_payload["keyframes"] = [
                {
                    "index": frame_index,
                    "timestamp": timestamps[frame_index] if frame_index < len(timestamps) else None,
                    "filename": image.name,
                    "url": f"/api/tasks/{task.task_id}/debug/keyframes/{index}/{image.name}",
                }
                for frame_index, image in enumerate(images)
            ]
            if payload.get("output") is not None:
                payload["output"]["keyframe_timestamps"] = timestamps
            payload["debug"] = debug_payload
            self._persist_debug_artifact(task.task_id, f"segment_{index:04d}_vision.json", payload)
            enriched_insights.append(payload)

        enhanced = apply_video_insights(result, enriched_insights, task.video_model or self.config.video_llm_model)
        if enhanced.video_enhancement_used:
            return enhanced
        summary = dict(enhanced.debug_summary)
        summary["video_enhancement"] = {
            "model": task.video_model or self.config.video_llm_model,
            "used": False,
            "skip_reason": "video model returned no usable description",
            "segment_count": len(enhanced.segments),
        }
        return enhanced.__class__(
            video_duration=enhanced.video_duration,
            total_frames_analyzed=enhanced.total_frames_analyzed,
            fps=enhanced.fps,
            width=enhanced.width,
            height=enhanced.height,
            segments=enhanced.segments,
            generated_at=enhanced.generated_at,
            result_source=enhanced.result_source,
            text_model=enhanced.text_model,
            video_model=task.video_model or self.config.video_llm_model,
            text_enhancement_used=enhanced.text_enhancement_used,
            video_enhancement_used=False,
            debug_available=enhanced.debug_available,
            debug_summary=summary,
            sampling_profile=enhanced.sampling_profile,
        )

    def _apply_text_enhancement(self, task, result: AnalysisResult) -> AnalysisResult:
        descriptions = self.description_enhancer.enhance(result.segments, model=task.text_model or self.config.llm_model)
        for index, payload in enumerate(descriptions):
            if payload is not None:
                self._persist_debug_artifact(task.task_id, f"segment_{index:04d}_text.json", payload)
        enhanced = apply_enhanced_descriptions(result, descriptions, text_model=task.text_model or self.config.llm_model)
        if not enhanced.text_enhancement_used:
            return mark_text_fallback(enhanced, "text model returned no usable description", task.text_model or self.config.llm_model)
        return enhanced

    def _merge_task_chunks(self, task_id: str, chunks: list[AnalysisChunk]) -> AnalysisResult:
        all_frames: dict[float, DetectionFrame] = {}
        track_summaries: list[dict] = []
        sampling_profiles: list[dict] = []
        for chunk in chunks:
            payload = self._load_chunk_payload(chunk)
            for item in payload.get("detections", []):
                frame = DetectionFrame.from_dict(item)
                all_frames[frame.timestamp] = frame
            track_summaries.extend(payload.get("track_summary", []))
            if payload.get("sampling_profile"):
                sampling_profiles.append(payload["sampling_profile"])

        task_media = self.repository.get_media_for_task(task_id, ArtifactKind.ORIGINAL_VIDEO)
        metadata = probe_video(Path(task_media[0].path))
        merged_sampling = self._merge_sampling_profiles(sampling_profiles)
        result = build_analysis_result(
            list(all_frames.values()),
            video_duration=metadata.duration_seconds,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            sampling_profile=merged_sampling,
        )
        result.result_source = "rule"
        result.debug_summary["tracker"] = {
            "track_count": len({item.get("track_id") for item in track_summaries if item.get("track_id") is not None}),
            "track_summary": track_summaries,
        }
        return result

    def _merge_sampling_profiles(self, profiles: list[dict]) -> dict:
        if not profiles:
            return {}
        return {
            "sampling_profile": "balanced_adaptive",
            "base_sample_interval": profiles[0].get("base_sample_interval"),
            "stable_person_interval": profiles[0].get("stable_person_interval"),
            "refined_sample_interval": profiles[0].get("refined_sample_interval"),
            "strong_refined_sample_interval": profiles[0].get("strong_refined_sample_interval"),
            "refined_ranges": [item for profile in profiles for item in profile.get("sampling_events", [])],
        }

    def _load_chunk_payload(self, chunk: AnalysisChunk) -> dict:
        if chunk.artifact_path and Path(chunk.artifact_path).exists():
            return json.loads(Path(chunk.artifact_path).read_text(encoding="utf-8"))
        if chunk.result_payload:
            return chunk.result_payload
        raise ValueError(f"missing chunk artifact: {chunk.chunk_id}")

    def _persist_result_artifacts(self, task_id: str, result: AnalysisResult) -> None:
        result_path = self.config.artifact_dir / task_id / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        existing_results = self.repository.get_media_for_task(task_id, ArtifactKind.RESULT_JSON)
        if not any(item.path == str(result_path) for item in existing_results):
            self.repository.create_media_file(
                file_id=str(uuid.uuid4()),
                task_id=task_id,
                kind=ArtifactKind.RESULT_JSON,
                path=str(result_path),
                original_name=result_path.name,
                mime_type="application/json",
                extension=".json",
                size_bytes=result_path.stat().st_size,
                sha256=f"result-{task_id}",
                ttl_seconds=None,
                metadata={"result_source": result.result_source},
            )

        source_video = self.repository.get_media_for_task(task_id, ArtifactKind.ORIGINAL_VIDEO)
        if not source_video:
            return
        video_path = Path(source_video[0].path)
        existing_thumbnails = self.repository.get_media_for_task(task_id, ArtifactKind.THUMBNAIL)
        existing_by_segment = {item.metadata.get("segment_index"): item for item in existing_thumbnails}
        for index, segment in enumerate(result.segments):
            thumbnail_path = self.config.artifact_dir / task_id / "thumbnails" / f"{index:04d}.jpg"
            if not thumbnail_path.exists():
                try:
                    save_thumbnail(video_path, segment.thumbnail_timestamp, thumbnail_path)
                except Exception:
                    continue
            existing = existing_by_segment.get(index)
            if existing and existing.path == str(thumbnail_path):
                continue
            self.repository.create_media_file(
                file_id=str(uuid.uuid4()),
                task_id=task_id,
                kind=ArtifactKind.THUMBNAIL,
                path=str(thumbnail_path),
                original_name=thumbnail_path.name,
                mime_type="image/jpeg",
                extension=".jpg",
                size_bytes=thumbnail_path.stat().st_size,
                sha256=f"thumbnail-{task_id}-{index}",
                ttl_seconds=None,
                metadata={
                    "segment_index": index,
                    "timestamp": segment.thumbnail_timestamp,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                },
            )

    def _persist_debug_artifact(self, task_id: str, filename: str, payload: dict) -> Path:
        debug_dir = self.config.artifact_dir / task_id / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _segment_keyframes(self, task_id: str, segment_index: int, timestamps: list[float] | None = None) -> list[dict]:
        frame_dir = self.config.artifact_dir / task_id / "keyframes" / f"{segment_index:04d}"
        if not frame_dir.exists():
            return []
        timestamps = timestamps or []
        payload: list[dict] = []
        for frame_index, path in enumerate(sorted(frame_dir.glob("*.jpg"))):
            payload.append(
                {
                    "index": frame_index,
                    "timestamp": timestamps[frame_index] if frame_index < len(timestamps) else None,
                    "filename": path.name,
                    "url": f"/api/tasks/{task_id}/debug/keyframes/{segment_index}/{path.name}",
                }
            )
        return payload

    def _load_result_and_segment(self, task_id: str, segment_index: int) -> tuple[AnalysisResult, object] | tuple[None, None]:
        result_payload = self.repository.get_result(task_id)
        if not result_payload:
            return None, None
        result = AnalysisResult.from_dict(result_payload)
        if segment_index < 0 or segment_index >= len(result.segments):
            return None, None
        return result, result.segments[segment_index]

    def _ordered_task_frames(self, task_id: str) -> list[DetectionFrame]:
        all_frames: dict[float, DetectionFrame] = {}
        for chunk in self.repository.list_chunks(task_id):
            payload = self._load_chunk_payload(chunk)
            for item in payload.get("detections", []):
                frame = DetectionFrame.from_dict(item)
                all_frames[frame.timestamp] = frame
        return [all_frames[key] for key in sorted(all_frames.keys())]

    def _build_segment_debug_payload(self, task_id: str, segment_index: int, result: AnalysisResult, segment, *, vision_debug: dict | None, text_debug: dict | None, clip: dict | None = None) -> dict:
        features = dict(segment.features)
        track_ids = list(features.get("track_ids", []))
        return {
            "task_id": task_id,
            "index": segment_index,
            "segment": {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "description": segment.description,
                "rule_description": segment.rule_description,
                "video_description": segment.video_description,
                "video_labels": segment.video_labels,
                "action": segment.action,
                "scene": segment.scene,
                "confidence": segment.confidence,
                "video_result_status": segment.video_result_status,
                "fallback_reason": segment.fallback_reason,
                "parse_mode": segment.parse_mode,
                "raw_response_present": segment.raw_response_present,
                "person_count_range": segment.person_count_range,
                "track_count": segment.track_count,
                "track_ids": track_ids,
                "scene_change_score": segment.scene_change_score,
                "sampling_events": segment.sampling_events,
            },
            "tracker": {
                "track_count": segment.track_count,
                "track_ids": track_ids,
                "person_count_range": segment.person_count_range,
                "scene_change_score": segment.scene_change_score,
                "sampling_events": segment.sampling_events,
            },
            "sampling_profile": result.sampling_profile,
            "keyframes": self._segment_keyframes(task_id, segment_index, segment.keyframe_timestamps),
            "clip": clip,
            "vision_debug": vision_debug,
            "text_debug": text_debug,
            "timings": {
                "vision_latency_ms": ((vision_debug or {}).get("debug") or {}).get("vision_latency_ms"),
                "text_latency_ms": ((text_debug or {}).get("debug") or {}).get("text_latency_ms"),
            },
        }

    def rerun_segment_debug(
        self,
        task_id: str,
        segment_index: int,
        *,
        mode: str,
        video_model: str | None,
        text_model: str | None,
        run_video: bool,
        run_text: bool,
    ) -> dict | None:
        result, segment = self._load_result_and_segment(task_id, segment_index)
        task = self.repository.get_task(task_id)
        if not result or not segment or not task:
            return None
        media = self.repository.get_media_for_task(task_id, ArtifactKind.ORIGINAL_VIDEO)
        if not media:
            return None
        video_path = Path(media[0].path)
        ordered_frames = self._ordered_task_frames(task_id)
        timestamps = segment.keyframe_timestamps or self._select_keyframe_timestamps(segment, ordered_frames)
        images_dir = self.config.artifact_dir / task_id / "debug" / f"segment_{segment_index:04d}_images"
        clip_path = self.config.artifact_dir / task_id / "debug" / f"segment_{segment_index:04d}_clip.mp4"
        images = extract_segment_keyframes(video_path, timestamps, images_dir, self.config.keyframe_max_size)
        clip_info = None
        if mode in {"clip", "both"}:
            export_segment_clip(video_path, segment.start_time, segment.end_time, clip_path)
            clip_info = {
                "path": str(clip_path),
                "url": f"/api/tasks/{task_id}/debug/clips/{segment_index}",
                "start_time": segment.start_time,
                "end_time": segment.end_time,
            }

        video_runs: dict[str, dict | None] = {}
        chosen_video = video_model or task.video_model or self.config.video_llm_model
        if run_video and mode in {"images", "both"}:
            payload = self.video_enhancer.analyze([segment], [[path for path, _ in images]], model=chosen_video)[0]
            self._persist_debug_artifact(task_id, f"segment_{segment_index:04d}_vision_images.json", payload or {})
            video_runs["images"] = payload
        if run_video and mode in {"clip", "both"}:
            clip_timestamps = [0.0, max((segment.end_time - segment.start_time) / 2, 0.1), max(segment.end_time - segment.start_time - 0.1, 0.1)]
            clip_images_dir = self.config.artifact_dir / task_id / "debug" / f"segment_{segment_index:04d}_clip_frames"
            clip_images = extract_segment_keyframes(clip_path, clip_timestamps, clip_images_dir, self.config.keyframe_max_size)
            payload = self.video_enhancer.analyze([segment], [[path for path, _ in clip_images]], model=chosen_video)[0]
            if payload:
                payload.setdefault("debug", {})["input_mode"] = "clip"
            self._persist_debug_artifact(task_id, f"segment_{segment_index:04d}_vision_clip.json", payload or {})
            video_runs["clip"] = payload

        text_runs: dict[str, dict | None] = {}
        chosen_text = text_model or task.text_model or self.config.llm_model
        if run_text and not video_runs:
            text_payload = self.description_enhancer.enhance([segment], model=chosen_text)[0]
            self._persist_debug_artifact(task_id, f"segment_{segment_index:04d}_text_from_rule.json", text_payload or {})
            text_runs["rule"] = text_payload
        if run_text:
            for run_name, vision_payload in video_runs.items():
                derived_segment = segment
                if vision_payload and vision_payload.get("output", {}).get("description"):
                    derived_segment = type(segment).from_dict(segment.to_dict())
                    derived_segment.video_description = vision_payload["output"]["description"]
                    derived_segment.description = vision_payload["output"]["description"]
                text_payload = self.description_enhancer.enhance([derived_segment], model=chosen_text)[0]
                self._persist_debug_artifact(task_id, f"segment_{segment_index:04d}_text_from_{run_name}.json", text_payload or {})
                text_runs[run_name] = text_payload

        primary_vision = video_runs.get("clip") or video_runs.get("images")
        primary_text = text_runs.get("clip") or text_runs.get("images")
        return self._build_segment_debug_payload(
            task_id,
            segment_index,
            result,
            segment,
            vision_debug=primary_vision,
            text_debug=primary_text,
            clip=clip_info,
        ) | {
            "rerun": {
                "mode": mode,
                "video_model": chosen_video if run_video else None,
                "text_model": chosen_text if run_text else None,
                "vision_runs": video_runs,
                "text_runs": text_runs,
            }
        }

    def get_task_debug(self, task_id: str) -> dict | None:
        task = self.repository.get_task(task_id)
        result_payload = self.repository.get_result(task_id)
        if not task or not result_payload:
            return None
        result = AnalysisResult.from_dict(result_payload)
        video_status_counts: dict[str, int] = {}
        text_status_counts: dict[str, int] = {}
        fallback_reason_counts: dict[str, int] = {}
        video_latencies: list[int] = []
        text_latencies: list[int] = []
        for index in range(len(result.segments)):
            segment_debug = self.get_segment_debug(task_id, index)
            vision = segment_debug.get("vision_debug") if segment_debug else None
            text = segment_debug.get("text_debug") if segment_debug else None
            status = (((vision or {}).get("debug") or {}).get("vision_status")) or result.segments[index].video_result_status or "unknown"
            video_status_counts[status] = video_status_counts.get(status, 0) + 1
            text_status = (((text or {}).get("debug") or {}).get("text_status")) or "not_called"
            text_status_counts[text_status] = text_status_counts.get(text_status, 0) + 1
            reason = result.segments[index].fallback_reason or (((vision or {}).get("debug") or {}).get("vision_fallback_reason"))
            if reason:
                fallback_reason_counts[str(reason)] = fallback_reason_counts.get(str(reason), 0) + 1
            video_latency = (((vision or {}).get("debug") or {}).get("vision_latency_ms"))
            if isinstance(video_latency, int):
                video_latencies.append(video_latency)
            text_latency = (((text or {}).get("debug") or {}).get("text_latency_ms"))
            if isinstance(text_latency, int):
                text_latencies.append(text_latency)
        return {
            "task_id": task_id,
            "status": task.status.value,
            "stage": task.stage,
            "progress": task.progress,
            "result_source": result.result_source,
            "sampling_profile": result.sampling_profile,
            "video_result_stats": result.debug_summary.get("video_result_stats", {}),
            "tracker": result.debug_summary.get("tracker", {}),
            "debug_summary": {
                **result.debug_summary,
                "video_status_counts": video_status_counts,
                "text_status_counts": text_status_counts,
                "fallback_reason_counts": fallback_reason_counts,
                "latency_summary": {
                    "avg_video_latency_ms": int(sum(video_latencies) / len(video_latencies)) if video_latencies else 0,
                    "max_video_latency_ms": max(video_latencies) if video_latencies else 0,
                    "avg_text_latency_ms": int(sum(text_latencies) / len(text_latencies)) if text_latencies else 0,
                    "max_text_latency_ms": max(text_latencies) if text_latencies else 0,
                },
            },
            "segments": [
                {
                    "index": index,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "rule_description": segment.rule_description,
                    "video_description": segment.video_description,
                    "description": segment.description,
                    "video_result_status": segment.video_result_status,
                    "fallback_reason": segment.fallback_reason,
                    "parse_mode": segment.parse_mode,
                    "raw_response_present": segment.raw_response_present,
                    "track_count": segment.track_count,
                    "person_count_range": segment.person_count_range,
                    "scene_change_score": segment.scene_change_score,
                    "sampling_events": segment.sampling_events,
                    "keyframes": self._segment_keyframes(task_id, index, segment.keyframe_timestamps),
                }
                for index, segment in enumerate(result.segments)
            ],
        }

    def get_segment_debug(self, task_id: str, segment_index: int) -> dict | None:
        result, segment = self._load_result_and_segment(task_id, segment_index)
        if not result or not segment:
            return None
        debug_dir = self.config.artifact_dir / task_id / "debug"
        vision_path = debug_dir / f"segment_{segment_index:04d}_vision.json"
        text_path = debug_dir / f"segment_{segment_index:04d}_text.json"
        clip_path = debug_dir / f"segment_{segment_index:04d}_clip.mp4"
        return self._build_segment_debug_payload(
            task_id,
            segment_index,
            result,
            segment,
            vision_debug=json.loads(vision_path.read_text(encoding="utf-8")) if vision_path.exists() else None,
            text_debug=json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else None,
            clip={
                "path": str(clip_path),
                "url": f"/api/tasks/{task_id}/debug/clips/{segment_index}",
                "start_time": segment.start_time,
                "end_time": segment.end_time,
            } if clip_path.exists() else None,
        ) | {"features": segment.features}
