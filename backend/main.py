"""FastAPI 后端入口"""
import asyncio
import json
import os
import uuid
from pathlib import Path
from dataclasses import asdict
from typing import Optional

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models import AnalysisTask, TaskStatus
from detector import PersonDetector
from analyzer import build_analysis_result

# ──────────── 配置 ────────────
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="监控视频人物活动检测")

# 静态文件
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# 任务存储（内存）
tasks: dict[str, AnalysisTask] = {}

# 全局检测器（懒加载）
_detector: Optional[PersonDetector] = None


def get_detector() -> PersonDetector:
    global _detector
    if _detector is None:
        _detector = PersonDetector()
    return _detector


# ──────────── 页面 ────────────

@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ──────────── API ────────────

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件"""
    if not file.filename:
        raise HTTPException(400, "缺少文件名")

    # 检查文件类型
    ext = Path(file.filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".ts"}
    if ext not in allowed:
        raise HTTPException(400, f"不支持的文件格式: {ext}")

    task_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{task_id}{ext}"

    # 流式保存大文件
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    task = AnalysisTask(task_id=task_id, video_path=str(save_path))
    tasks[task_id] = task

    return {"task_id": task_id, "filename": file.filename}


@app.post("/api/analyze/{task_id}")
async def start_analysis(task_id: str):
    """开始分析任务"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task.status == TaskStatus.PROCESSING:
        raise HTTPException(400, "任务正在处理中")

    task.status = TaskStatus.PROCESSING
    task.progress = 0

    # 在后台线程中运行（不阻塞事件循环）
    asyncio.get_event_loop().run_in_executor(None, _run_analysis, task_id)

    return {"task_id": task_id, "status": "processing"}


def _run_analysis(task_id: str):
    """在后台线程中执行视频分析"""
    task = tasks.get(task_id)
    if not task:
        return

    try:
        detector = get_detector()

        def on_progress(pct: float, msg: str):
            task.progress = pct
            task.progress_message = msg

        detections, video_info = detector.process_video(
            task.video_path,
            sample_interval=1.0,
            progress_callback=on_progress
        )

        task.progress = 85
        task.progress_message = "正在分析活动模式..."

        result = build_analysis_result(detections, video_info)

        task.result = result
        task.progress = 100
        task.progress_message = f"分析完成，发现 {len(result.segments)} 段人物活动"
        task.status = TaskStatus.COMPLETED

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.progress_message = f"分析失败: {e}"


@app.get("/api/status/{task_id}")
async def get_status_sse(task_id: str):
    """SSE 推送分析进度"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    async def event_stream():
        while True:
            data = {
                "status": task.status.value,
                "progress": task.progress,
                "message": task.progress_message,
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """获取分析结果"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task.status != TaskStatus.COMPLETED or not task.result:
        raise HTTPException(400, "分析尚未完成")

    result = task.result
    segments_data = []
    for seg in result.segments:
        segments_data.append({
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "duration": seg.duration,
            "max_persons": seg.max_persons,
            "description": seg.description,
            "thumbnail_timestamp": seg.thumbnail_timestamp,
        })

    return {
        "video_duration": result.video_duration,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "total_frames_analyzed": result.total_frames_analyzed,
        "total_segments": len(segments_data),
        "segments": segments_data,
    }


@app.get("/api/frame/{task_id}/{timestamp}")
async def get_frame(task_id: str, timestamp: float):
    """获取视频指定时间点的帧截图"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    cap = cv2.VideoCapture(task.video_path)
    if not cap.isOpened():
        raise HTTPException(500, "无法打开视频文件")

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, "无法读取该时间点的帧")

    # 编码为JPEG
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@app.get("/api/video/{task_id}")
async def serve_video(task_id: str):
    """提供视频文件流式播放"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    video_path = Path(task.video_path)
    if not video_path.exists():
        raise HTTPException(404, "视频文件不存在")

    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        filename=video_path.name
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
