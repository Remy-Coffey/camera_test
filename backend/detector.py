"""YOLO人物检测模块"""
import cv2
import numpy as np
from ultralytics import YOLO
from models import BBox, FrameDetection
from typing import Callable, Optional


class PersonDetector:
    """基于YOLOv8的人物检测器"""

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.4):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        # COCO数据集中 class 0 = person
        self.person_class_id = 0

    def detect_frame(self, frame: np.ndarray) -> list[BBox]:
        """检测单帧中的人物"""
        results = self.model(frame, verbose=False, classes=[self.person_class_id])
        persons = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    persons.append(BBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf
                    ))
        return persons

    def process_video(
        self,
        video_path: str,
        sample_interval: float = 1.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple[list[FrameDetection], dict]:
        """
        处理整个视频，按采样间隔抽帧检测。

        Args:
            video_path: 视频文件路径
            sample_interval: 采样间隔（秒）
            progress_callback: 进度回调 (progress_percent, message)

        Returns:
            (帧检测结果列表, 视频元信息dict)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # 计算采样帧间隔
        frame_interval = max(1, int(fps * sample_interval))
        total_samples = total_frames // frame_interval

        detections: list[FrameDetection] = []
        frame_index = 0
        sample_count = 0

        if progress_callback:
            progress_callback(0, f"开始分析视频 (时长: {duration:.1f}s, 共{total_samples}帧待分析)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp = frame_index / fps
                persons = self.detect_frame(frame)
                detections.append(FrameDetection(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    persons=persons
                ))
                sample_count += 1

                if progress_callback and total_samples > 0:
                    pct = (sample_count / total_samples) * 80  # 检测占80%进度
                    msg = f"检测中... {sample_count}/{total_samples} 帧"
                    if persons:
                        msg += f" (发现 {len(persons)} 人)"
                    progress_callback(pct, msg)

            frame_index += 1

        cap.release()

        video_info = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration,
        }

        return detections, video_info
