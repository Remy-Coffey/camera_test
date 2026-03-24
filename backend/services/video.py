from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import cv2


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(slots=True)
class VideoMetadata:
    path: Path
    size_bytes: int
    sha256: str
    mime_type: str | None
    extension: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {extension or 'unknown'}")
    return extension


def probe_video(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if total_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError("视频元数据无效，文件可能已损坏")
    duration_seconds = total_frames / fps if fps > 0 else 0.0
    return VideoMetadata(
        path=path,
        size_bytes=path.stat().st_size,
        sha256=compute_sha256(path),
        mime_type=mimetypes.guess_type(path.name)[0],
        extension=path.suffix.lower(),
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def _read_frame_at(video_path: Path, timestamp: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("无法读取指定时间点的视频帧")
    return frame


def save_thumbnail(video_path: Path, timestamp: float, output_path: Path) -> None:
    frame = _read_frame_at(video_path, timestamp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("缩略图编码失败")
    output_path.write_bytes(buffer.tobytes())


def extract_segment_keyframes(
    video_path: Path,
    timestamps: list[float],
    output_dir: Path,
    max_size: int,
) -> list[tuple[Path, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes: list[tuple[Path, float]] = []
    for index, timestamp in enumerate(timestamps):
        frame = _read_frame_at(video_path, timestamp)
        height, width = frame.shape[:2]
        scale = min(1.0, max_size / max(width, height))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        output_path = output_dir / f"{index:02d}.jpg"
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not success:
            continue
        output_path.write_bytes(buffer.tobytes())
        keyframes.append((output_path, timestamp))
    return keyframes


def export_segment_clip(video_path: Path, start_time: float, end_time: float, output_path: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 5.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_time) * 1000)

    try:
        while True:
            current_ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if current_ts > end_time:
                break
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return output_path
