"""数据模型定义"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BBox:
    """人物检测框"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FrameDetection:
    """单帧检测结果"""
    frame_index: int
    timestamp: float  # 秒
    persons: list[BBox] = field(default_factory=list)

    @property
    def has_person(self) -> bool:
        return len(self.persons) > 0


@dataclass
class ActivitySegment:
    """一段人物活动"""
    start_time: float  # 秒
    end_time: float
    max_persons: int
    description: str
    thumbnail_timestamp: float  # 用于缩略图的关键帧时间
    frames: list[FrameDetection] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AnalysisResult:
    """完整分析结果"""
    video_duration: float
    total_frames_analyzed: int
    segments: list[ActivitySegment] = field(default_factory=list)
    fps: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class AnalysisTask:
    """分析任务"""
    task_id: str
    video_path: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0-100
    progress_message: str = ""
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
