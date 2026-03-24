"""活动分析 & 描述生成模块"""
from models import FrameDetection, ActivitySegment, AnalysisResult


def _format_time(seconds: float) -> str:
    """格式化秒数为 MM:SS 或 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _describe_position(cx: float, cy: float, frame_w: int, frame_h: int) -> str:
    """描述人物在画面中的位置"""
    # 将画面分为3x3九宫格
    x_ratio = cx / frame_w
    y_ratio = cy / frame_h

    if x_ratio < 0.33:
        h_pos = "左侧"
    elif x_ratio < 0.66:
        h_pos = "中部"
    else:
        h_pos = "右侧"

    if y_ratio < 0.33:
        v_pos = "上方"
    elif y_ratio < 0.66:
        v_pos = "中间"
    else:
        v_pos = "下方"

    if h_pos == "中部" and v_pos == "中间":
        return "画面中央"
    if v_pos == "中间":
        return f"画面{h_pos}"
    if h_pos == "中部":
        return f"画面{v_pos}"
    return f"画面{v_pos}{h_pos}"


def _describe_direction(dx: float, dy: float) -> str:
    """根据位移向量描述运动方向"""
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return ""

    if abs(dx) > abs(dy) * 2:
        return "从左向右移动" if dx > 0 else "从右向左移动"
    elif abs(dy) > abs(dx) * 2:
        return "向下方移动" if dy > 0 else "向上方移动"
    else:
        h = "右" if dx > 0 else "左"
        v = "下" if dy > 0 else "上"
        return f"向{v}{h}方向移动"


def _generate_description(
    segment_frames: list[FrameDetection],
    frame_w: int,
    frame_h: int
) -> str:
    """为一段活动生成文字描述"""
    if not segment_frames:
        return "检测到人物活动"

    # 统计最大同时人数
    max_persons = max(len(f.persons) for f in segment_frames)
    duration = segment_frames[-1].timestamp - segment_frames[0].timestamp

    parts = []

    # — 人数
    if max_persons == 1:
        parts.append("1人")
    else:
        parts.append(f"最多{max_persons}人")

    # — 分析主要人物的轨迹（取每帧中面积最大的人物作为主人物）
    centers = []
    for f in segment_frames:
        if f.persons:
            main_person = max(f.persons, key=lambda b: b.area)
            centers.append(main_person.center)

    if len(centers) >= 2:
        # 起始和结束位置
        start_pos = _describe_position(centers[0][0], centers[0][1], frame_w, frame_h)
        end_pos = _describe_position(centers[-1][0], centers[-1][1], frame_w, frame_h)

        # 总位移
        total_dx = (centers[-1][0] - centers[0][0]) / frame_w
        total_dy = (centers[-1][1] - centers[0][1]) / frame_h

        # 计算运动幅度（归一化）
        movement = (total_dx ** 2 + total_dy ** 2) ** 0.5

        if movement < 0.08:
            # 基本静止
            parts.append(f"在{start_pos}区域停留")
            if duration > 3:
                parts.append(f"约{duration:.0f}秒")
        else:
            # 有明显移动
            direction = _describe_direction(total_dx, total_dy)
            if direction:
                parts.append(direction)
            if start_pos != end_pos:
                parts.append(f"从{start_pos}到{end_pos}")
    elif len(centers) == 1:
        pos = _describe_position(centers[0][0], centers[0][1], frame_w, frame_h)
        parts.append(f"出现在{pos}")

    # — 出入画面检测
    first_frame = segment_frames[0]
    last_frame = segment_frames[-1]

    # 检查是否从画面边缘进入
    if first_frame.persons:
        p = max(first_frame.persons, key=lambda b: b.area)
        if p.x1 < frame_w * 0.05 or p.x2 > frame_w * 0.95:
            parts.insert(1, "从画面边缘进入")
    if last_frame.persons:
        p = max(last_frame.persons, key=lambda b: b.area)
        if p.x1 < frame_w * 0.05 or p.x2 > frame_w * 0.95:
            if "进入" not in " ".join(parts):
                parts.append("离开画面")

    return "，".join(parts)


def merge_segments(
    detections: list[FrameDetection],
    gap_threshold: float = 3.0,
    min_segment_duration: float = 0.5,
    frame_w: int = 1920,
    frame_h: int = 1080
) -> list[ActivitySegment]:
    """
    将检测结果合并为活动片段。

    Args:
        detections: 按时间排序的帧检测结果
        gap_threshold: 无人帧间隔超过此值(秒)则分割为新片段
        min_segment_duration: 最短片段时长(秒)，过短的忽略
        frame_w: 视频宽度
        frame_h: 视频高度

    Returns:
        活动片段列表
    """
    if not detections:
        return []

    segments: list[ActivitySegment] = []
    current_frames: list[FrameDetection] = []
    last_person_time: float = -999

    for det in detections:
        if det.has_person:
            # 如果和上一次有人帧间隔太大，先保存当前片段
            if current_frames and (det.timestamp - last_person_time) > gap_threshold:
                seg = _build_segment(current_frames, frame_w, frame_h)
                if seg and seg.duration >= min_segment_duration:
                    segments.append(seg)
                current_frames = []

            current_frames.append(det)
            last_person_time = det.timestamp

    # 处理最后一段
    if current_frames:
        seg = _build_segment(current_frames, frame_w, frame_h)
        if seg and seg.duration >= min_segment_duration:
            segments.append(seg)

    return segments


def _build_segment(
    frames: list[FrameDetection],
    frame_w: int,
    frame_h: int
) -> ActivitySegment | None:
    """从帧列表构建活动片段"""
    if not frames:
        return None

    person_frames = [f for f in frames if f.has_person]
    if not person_frames:
        return None

    max_persons = max(len(f.persons) for f in person_frames)

    # 选择人数最多的帧作为缩略图
    best_frame = max(person_frames, key=lambda f: len(f.persons))

    description = _generate_description(person_frames, frame_w, frame_h)

    return ActivitySegment(
        start_time=person_frames[0].timestamp,
        end_time=person_frames[-1].timestamp,
        max_persons=max_persons,
        description=description,
        thumbnail_timestamp=best_frame.timestamp,
        frames=person_frames
    )


def build_analysis_result(
    detections: list[FrameDetection],
    video_info: dict,
    gap_threshold: float = 3.0
) -> AnalysisResult:
    """构建完整分析结果"""
    frame_w = video_info.get("width", 1920)
    frame_h = video_info.get("height", 1080)

    segments = merge_segments(
        detections,
        gap_threshold=gap_threshold,
        frame_w=frame_w,
        frame_h=frame_h
    )

    return AnalysisResult(
        video_duration=video_info.get("duration", 0),
        total_frames_analyzed=len(detections),
        segments=segments,
        fps=video_info.get("fps", 0),
        width=frame_w,
        height=frame_h
    )
