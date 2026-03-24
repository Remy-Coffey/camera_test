from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig
from domain import ArtifactKind
from repositories import SQLiteRepository
from services.cleanup import CleanupService
from services.orchestrator import TaskOrchestrator
from services.video import probe_video, validate_extension


class StartTaskRequest(BaseModel):
    llm_enabled: bool | None = None
    video_enhancement_enabled: bool | None = None
    text_model: str | None = None
    video_model: str | None = None
    performance_profile: str | None = None


class RerunSegmentRequest(BaseModel):
    mode: str = "images"
    video_model: str | None = None
    text_model: str | None = None
    run_video: bool = True
    run_text: bool = False
    performance_profile: str | None = None


def task_to_dict(task) -> dict:
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "stage": task.stage,
        "error": task.error,
        "chunk_count": task.chunk_count,
        "completed_chunks": task.completed_chunks,
        "processing_chunks": task.processing_chunks,
        "queued_chunks": task.queued_chunks,
        "llm_enabled": task.llm_enabled,
        "video_enhancement_enabled": task.video_enhancement_enabled,
        "text_model": task.text_model,
        "video_model": task.video_model,
        "performance_profile": getattr(task, "performance_profile", "balanced"),
        "recovery_stage": task.recovery_stage,
        "recovery_reason": task.recovery_reason,
        "last_recovered_at": task.last_recovered_at.isoformat() if task.last_recovered_at else None,
        "artifact_health": task.artifact_health,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


def create_app(config: AppConfig | None = None) -> FastAPI:
    app_config = config or AppConfig.from_env()
    app_config.ensure_directories()
    repository = SQLiteRepository(app_config.database_path)
    orchestrator = TaskOrchestrator(repository, app_config)
    cleanup_service = CleanupService(repository)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        orchestrator.start()
        cleanup_service.start()
        yield
        cleanup_service.stop()
        orchestrator.stop()

    app = FastAPI(title="巡影", lifespan=lifespan)
    app.state.config = app_config
    app.state.repository = repository
    app.state.orchestrator = orchestrator
    app.state.cleanup_service = cleanup_service
    app.mount("/static", StaticFiles(directory=str(app_config.frontend_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(app_config.frontend_dir / "index.html"))

    @app.get("/api/system/llm")
    async def llm_status():
        return JSONResponse(orchestrator.llm_status())

    @app.get("/api/system/models")
    async def models_status():
        return JSONResponse(orchestrator.models_status())

    @app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)):
        if not file.filename:
            raise HTTPException(status_code=400, detail="缺少文件名")
        try:
            extension = validate_extension(file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        task_id = uuid.uuid4().hex[:12]
        file_id = uuid.uuid4().hex
        save_path = app_config.upload_dir / f"{task_id}{extension}"
        written = 0

        with save_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > app_config.upload_max_bytes:
                    handle.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="文件过大，超过系统限制")
                handle.write(chunk)

        try:
            metadata = probe_video(save_path)
        except ValueError as exc:
            save_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        repository.create_task(
            task_id=task_id,
            video_file_id=file_id,
            llm_enabled=app_config.llm_enabled,
            video_enhancement_enabled=app_config.video_llm_enabled,
            text_model=app_config.llm_model,
            video_model=app_config.video_llm_model,
            performance_profile=app_config.default_performance_profile,
        )
        repository.create_media_file(
            file_id=file_id,
            task_id=task_id,
            kind=ArtifactKind.ORIGINAL_VIDEO,
            path=str(save_path),
            original_name=file.filename,
            mime_type=file.content_type or metadata.mime_type,
            extension=extension,
            size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
            ttl_seconds=app_config.raw_video_ttl_seconds,
            metadata={
                "fps": metadata.fps,
                "duration_seconds": metadata.duration_seconds,
                "width": metadata.width,
                "height": metadata.height,
                "size_bytes": metadata.size_bytes,
            },
        )
        task = repository.get_task(task_id)
        return {"task_id": task_id, "task": task_to_dict(task), "video": repository.get_media_file(file_id).metadata}

    @app.post("/api/tasks/{task_id}/start")
    async def start_task(task_id: str, options: StartTaskRequest | None = None):
        task = repository.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.status.value in {"processing", "merging", "enhancing", "completed"}:
            return {"task": task_to_dict(task)}
        payload = options or StartTaskRequest()
        try:
            result = orchestrator.prepare_task(
                task_id,
                llm_enabled=payload.llm_enabled,
                video_enhancement_enabled=payload.video_enhancement_enabled,
                text_model=payload.text_model,
                video_model=payload.video_model,
                performance_profile=payload.performance_profile,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        task = repository.get_task(task_id)
        return {"task_id": result.task_id, "chunk_count": result.chunk_count, "task": task_to_dict(task)}

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str):
        task = repository.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        try:
            orchestrator.cancel_task(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"task": task_to_dict(repository.get_task(task_id))}

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        task = repository.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        media = repository.get_media_for_task(task_id, ArtifactKind.ORIGINAL_VIDEO)
        return {"task": task_to_dict(task), "video": media[0].metadata if media else {}}

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str):
        if not repository.get_task(task_id):
            raise HTTPException(status_code=404, detail="任务不存在")

        async def event_stream():
            while True:
                task = repository.get_task(task_id)
                if not task:
                    break
                yield f"data: {json.dumps({'task': task_to_dict(task)}, ensure_ascii=False)}\n\n"
                if task.status.value in {"completed", "cancelled", "failed", "expired"}:
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/tasks/{task_id}/result")
    async def get_task_result(task_id: str):
        task = repository.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        result = repository.get_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="结果尚未生成")
        return JSONResponse(result)

    @app.get("/api/tasks/{task_id}/debug")
    async def get_task_debug(task_id: str):
        if not repository.get_task(task_id):
            raise HTTPException(status_code=404, detail="任务不存在")
        payload = orchestrator.get_task_debug(task_id)
        if not payload:
            raise HTTPException(status_code=404, detail="调试信息尚未生成")
        return JSONResponse(payload)

    @app.get("/api/tasks/{task_id}/debug/segments/{segment_index}")
    async def get_segment_debug(task_id: str, segment_index: int):
        if not repository.get_task(task_id):
            raise HTTPException(status_code=404, detail="任务不存在")
        payload = orchestrator.get_segment_debug(task_id, segment_index)
        if not payload:
            raise HTTPException(status_code=404, detail="片段调试信息不存在")
        return JSONResponse(payload)

    @app.post("/api/tasks/{task_id}/debug/segments/{segment_index}/rerun")
    async def rerun_segment_debug(task_id: str, segment_index: int, options: RerunSegmentRequest):
        if not repository.get_task(task_id):
            raise HTTPException(status_code=404, detail="任务不存在")
        if options.mode not in {"images", "clip", "both"}:
            raise HTTPException(status_code=400, detail="mode 仅支持 images / clip / both")
        payload = orchestrator.rerun_segment_debug(
            task_id,
            segment_index,
            mode=options.mode,
            video_model=options.video_model,
            text_model=options.text_model,
            run_video=options.run_video,
            run_text=options.run_text,
            performance_profile=options.performance_profile,
        )
        if not payload:
            raise HTTPException(status_code=404, detail="片段调试信息不存在")
        return JSONResponse(payload)

    @app.get("/api/tasks/{task_id}/debug/keyframes/{segment_index}/{filename}")
    async def get_debug_keyframe(task_id: str, segment_index: int, filename: str):
        path = app_config.artifact_dir / task_id / "keyframes" / f"{segment_index:04d}" / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="关键帧不存在")
        return FileResponse(str(path), media_type="image/jpeg")

    @app.get("/api/tasks/{task_id}/debug/clips/{segment_index}")
    async def get_debug_clip(task_id: str, segment_index: int):
        path = app_config.artifact_dir / task_id / "debug" / f"segment_{segment_index:04d}_clip.mp4"
        if not path.exists():
            raise HTTPException(status_code=404, detail="调试片段不存在")
        return FileResponse(str(path), media_type="video/mp4")

    @app.get("/api/tasks/{task_id}/artifacts/{kind}")
    async def get_artifact(task_id: str, kind: str, request: Request):
        try:
            artifact_kind = ArtifactKind(kind)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="未知 artifact 类型") from exc

        files = repository.get_media_for_task(task_id, artifact_kind)
        if artifact_kind == ArtifactKind.THUMBNAIL:
            segment_index = request.query_params.get("segment_index")
            if segment_index is None:
                raise HTTPException(status_code=400, detail="thumbnail 需要 segment_index 参数")
            target = next((item for item in files if item.metadata.get("segment_index") == int(segment_index)), None)
            if not target:
                raise HTTPException(status_code=404, detail="缩略图不存在")
            return FileResponse(target.path, media_type=target.mime_type or "image/jpeg")

        if artifact_kind == ArtifactKind.CHUNK_RESULT:
            chunk_id = request.query_params.get("chunk_id")
            if chunk_id is None:
                raise HTTPException(status_code=400, detail="chunk_result 需要 chunk_id 参数")
            target = next((item for item in files if item.metadata.get("chunk_id") == chunk_id), None)
            if not target:
                raise HTTPException(status_code=404, detail="chunk artifact 不存在")
            return FileResponse(target.path, media_type=target.mime_type or "application/json")

        if not files:
            raise HTTPException(status_code=404, detail="artifact 不存在")
        return FileResponse(files[0].path, media_type=files[0].mime_type)

    return app
