from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from domain import ActivitySegment, AnalysisResult, DetectionFrame


def describe_position(cx: float, cy: float, frame_w: int, frame_h: int) -> str:
    x_ratio = cx / max(frame_w, 1)
    y_ratio = cy / max(frame_h, 1)

    if x_ratio < 0.33:
        horizontal = "左侧"
    elif x_ratio < 0.66:
        horizontal = "中部"
    else:
        horizontal = "右侧"

    if y_ratio < 0.33:
        vertical = "上方"
    elif y_ratio < 0.66:
        vertical = "中间"
    else:
        vertical = "下方"

    if horizontal == "中部" and vertical == "中间":
        return "画面中央"
    if vertical == "中间":
        return f"画面{horizontal}"
    if horizontal == "中部":
        return f"画面{vertical}"
    return f"画面{vertical}{horizontal}"


def describe_direction(dx: float, dy: float) -> str:
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return "基本停留在原地"
    if abs(dx) > abs(dy) * 2:
        return "从左向右移动" if dx > 0 else "从右向左移动"
    if abs(dy) > abs(dx) * 2:
        return "向下移动" if dy > 0 else "向上移动"
    horizontal = "右" if dx > 0 else "左"
    vertical = "下" if dy > 0 else "上"
    return f"向{vertical}{horizontal}方向移动"


def _primary_boxes(frames: list[DetectionFrame]):
    return [max(frame.persons, key=lambda box: box.area) for frame in frames if frame.persons]


def _track_set(frame: DetectionFrame) -> set[int]:
    return {track_id for track_id in frame.track_ids if track_id is not None}


def _track_overlap(left: DetectionFrame, right: DetectionFrame) -> float:
    left_tracks = _track_set(left)
    right_tracks = _track_set(right)
    if not left_tracks and not right_tracks:
        return 1.0
    if not left_tracks or not right_tracks:
        return 0.0
    return len(left_tracks & right_tracks) / len(left_tracks | right_tracks)


def _main_track_changed(previous: DetectionFrame, current: DetectionFrame) -> bool:
    if not previous.track_ids or not current.track_ids:
        return False
    return previous.track_ids[0] != current.track_ids[0]


def _sampling_events(frames: list[DetectionFrame]) -> list[str]:
    events: list[str] = []
    if any(frame.sampling_mode == "refined" for frame in frames):
        events.append("变化区域已自动加密抽帧")
    if any(frame.scene_changed for frame in frames):
        events.append("片段中存在明显画面变化")
    if len({frame.person_count for frame in frames}) > 1:
        events.append("片段中人数发生变化")
    return events


def _should_split(previous: DetectionFrame, current: DetectionFrame, gap_threshold: float) -> bool:
    if current.timestamp - previous.timestamp > gap_threshold:
        return True
    if previous.person_count != current.person_count:
        return True
    if current.scene_changed and current.scene_change_score >= 0.22:
        return True
    if _main_track_changed(previous, current):
        return True
    if previous.has_person and current.has_person and _track_overlap(previous, current) < 0.3:
        return True
    return False


def build_segment_features(
    frames: list[DetectionFrame],
    frame_w: int,
    frame_h: int,
) -> tuple[dict[str, Any], str]:
    if not frames:
        return {"max_persons": 0, "movement": "unknown"}, "检测到人物活动"

    person_counts = [frame.person_count or len(frame.persons) for frame in frames]
    max_persons = max(person_counts)
    min_persons = min(person_counts)
    primary_boxes = _primary_boxes(frames)
    duration = max(0.0, frames[-1].timestamp - frames[0].timestamp)
    track_ids = sorted({track_id for frame in frames for track_id in frame.track_ids if track_id is not None})
    scene_change_score = max((frame.scene_change_score for frame in frames), default=0.0)

    if primary_boxes:
        centers = [box.center for box in primary_boxes]
        start_pos = describe_position(centers[0][0], centers[0][1], frame_w, frame_h)
        end_pos = describe_position(centers[-1][0], centers[-1][1], frame_w, frame_h)
        dx = (centers[-1][0] - centers[0][0]) / max(frame_w, 1)
        dy = (centers[-1][1] - centers[0][1]) / max(frame_h, 1)
        movement_score = (dx**2 + dy**2) ** 0.5
        movement_kind = "stationary" if movement_score < 0.08 else "moving"
        direction = describe_direction(dx, dy)
    else:
        start_pos = "画面中央"
        end_pos = "画面中央"
        movement_kind = "unknown"
        direction = "动作不明确"

    features = {
        "max_persons": max_persons,
        "person_count_range": [min_persons, max_persons],
        "duration_seconds": round(duration, 2),
        "start_position": start_pos,
        "end_position": end_pos,
        "movement": movement_kind,
        "direction": direction,
        "track_count": len(track_ids),
        "track_ids": track_ids,
        "scene_change_score": round(scene_change_score, 4),
        "scene_changed": scene_change_score >= 0.22,
        "count_changed": min_persons != max_persons,
        "sampling_modes": sorted({frame.sampling_mode for frame in frames}),
    }

    parts: list[str] = []
    if min_persons == max_persons:
        parts.append(f"画面中持续有{max_persons}人")
    else:
        parts.append(f"画面中人数在{min_persons}到{max_persons}人之间变化")

    if movement_kind == "stationary":
        parts.append(f"主体主要在{start_pos}附近停留")
    elif movement_kind == "moving":
        if start_pos != end_pos:
            parts.append(f"主体{direction}，从{start_pos}移动到{end_pos}")
        else:
            parts.append(f"主体在{start_pos}附近{direction}")

    if len(track_ids) > max_persons:
        parts.append("期间出现了人物切换或多人轮流进入画面")
    if scene_change_score >= 0.22:
        parts.append("这一段伴随明显画面变化")
    if duration >= 5 and movement_kind == "stationary":
        parts.append(f"停留约{duration:.0f}秒")

    return features, "，".join(parts)


def build_segment(frames: list[DetectionFrame], frame_w: int, frame_h: int) -> ActivitySegment | None:
    if not frames:
        return None
    person_frames = [frame for frame in frames if frame.has_person]
    if not person_frames:
        return None
    features, rule_description = build_segment_features(person_frames, frame_w, frame_h)
    best_frame = max(person_frames, key=lambda frame: (frame.person_count, len(frame.track_ids)))
    return ActivitySegment(
        start_time=person_frames[0].timestamp,
        end_time=person_frames[-1].timestamp,
        max_persons=max(frame.person_count for frame in person_frames),
        description=rule_description,
        rule_description=rule_description,
        thumbnail_timestamp=best_frame.timestamp,
        features=features,
        person_count_range=list(features.get("person_count_range", [])),
        track_count=int(features.get("track_count", 0)),
        scene_change_score=float(features.get("scene_change_score", 0.0)),
        sampling_events=_sampling_events(person_frames),
    )


def merge_segments(
    detections: list[DetectionFrame],
    gap_threshold: float,
    frame_w: int,
    frame_h: int,
) -> list[ActivitySegment]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.timestamp)
    segments: list[ActivitySegment] = []
    current: list[DetectionFrame] = []
    previous_person_frame: DetectionFrame | None = None

    for frame in ordered:
        if not frame.has_person:
            continue
        if current and previous_person_frame and _should_split(previous_person_frame, frame, gap_threshold):
            segment = build_segment(current, frame_w, frame_h)
            if segment:
                segments.append(segment)
            current = []
        current.append(frame)
        previous_person_frame = frame

    if current:
        segment = build_segment(current, frame_w, frame_h)
        if segment:
            segments.append(segment)
    return segments


def build_analysis_result(
    detections: list[DetectionFrame],
    *,
    video_duration: float,
    fps: float,
    width: int,
    height: int,
    gap_threshold: float = 3.0,
    sampling_profile: dict[str, Any] | None = None,
) -> AnalysisResult:
    segments = merge_segments(detections, gap_threshold=gap_threshold, frame_w=width, frame_h=height)
    video_result_stats = {
        "success": 0,
        "weak_success": 0,
        "fallback": len(segments),
    }
    return AnalysisResult(
        video_duration=video_duration,
        total_frames_analyzed=len(detections),
        fps=fps,
        width=width,
        height=height,
        segments=segments,
        generated_at=datetime.now(timezone.utc),
        debug_summary={
            "segmentation": {
                "gap_threshold": gap_threshold,
                "frame_count": len(detections),
                "segment_count": len(segments),
            },
            "video_result_stats": video_result_stats,
        },
        sampling_profile=sampling_profile or {},
    )


def apply_video_insights(
    result: AnalysisResult,
    insights: list[dict[str, Any] | None],
    video_model: str | None,
) -> AnalysisResult:
    segments: list[ActivitySegment] = []
    used = False
    weak_used = False
    debug_available = result.debug_available
    success_count = 0
    weak_count = 0
    fallback_count = 0

    for segment, insight in zip(result.segments, insights):
        if not insight:
            fallback_count += 1
            segments.append(segment)
            continue

        output = insight.get("output") if isinstance(insight, dict) and "output" in insight else insight
        debug_payload = dict(insight.get("debug", {})) if isinstance(insight, dict) else {}
        fallback_reason = insight.get("fallback_reason") if isinstance(insight, dict) else None
        parse_mode = insight.get("parse_mode") if isinstance(insight, dict) else None
        raw_response_present = bool(insight.get("raw_response_present")) if isinstance(insight, dict) else False
        result_status = str(insight.get("video_result_status", "fallback")) if isinstance(insight, dict) else "fallback"

        if not output or not str(output.get("description", "")).strip():
            fallback_count += 1
            segments.append(
                replace(
                    segment,
                    debug_available=segment.debug_available or bool(debug_payload),
                    fallback_reason=fallback_reason or segment.fallback_reason,
                    video_result_status=result_status,
                    parse_mode=parse_mode,
                    raw_response_present=raw_response_present,
                    debug={**segment.debug, **debug_payload} if debug_payload else dict(segment.debug),
                )
            )
            debug_available = debug_available or bool(debug_payload)
            continue

        description = str(output.get("description", "")).strip()
        labels = [str(item) for item in output.get("labels", []) if str(item).strip()]
        features = dict(segment.features)
        if labels:
            features["video_labels"] = labels
        if output.get("action"):
            features["video_action"] = str(output["action"]).strip()
        if output.get("scene"):
            features["video_scene"] = str(output["scene"]).strip()
        confidence = float(output["confidence"]) if output.get("confidence") is not None else None

        used = used or result_status in {"success", "weak_success"}
        weak_used = weak_used or result_status == "weak_success"
        if result_status == "weak_success":
            weak_count += 1
        else:
            success_count += 1

        segments.append(
            replace(
                segment,
                description=description,
                video_description=description,
                video_labels=labels,
                keyframe_timestamps=[float(item) for item in output.get("keyframe_timestamps", [])],
                action=str(output.get("action", "")).strip() or None,
                scene=str(output.get("scene", "")).strip() or None,
                confidence=confidence,
                features=features,
                debug_available=segment.debug_available or bool(debug_payload),
                fallback_reason=None,
                video_result_status=result_status,
                parse_mode=parse_mode,
                raw_response_present=raw_response_present,
                debug={**segment.debug, **debug_payload} if debug_payload else dict(segment.debug),
            )
        )
        debug_available = debug_available or bool(debug_payload)

    summary = dict(result.debug_summary)
    summary["video_enhancement"] = {
        "model": video_model,
        "used": used,
        "weak_success_used": weak_used,
        "segment_count": len(result.segments),
    }
    summary["video_result_stats"] = {
        "success": success_count,
        "weak_success": weak_count,
        "fallback": fallback_count,
    }
    result_source = "video_llm" if used else result.result_source
    return replace(
        result,
        segments=segments,
        result_source=result_source,
        video_model=video_model,
        video_enhancement_used=used,
        debug_available=debug_available,
        debug_summary=summary,
    )


def apply_enhanced_descriptions(
    result: AnalysisResult,
    descriptions: list[dict[str, Any] | None],
    text_model: str | None = None,
) -> AnalysisResult:
    segments: list[ActivitySegment] = []
    used = False
    debug_available = result.debug_available

    for segment, enhanced in zip(result.segments, descriptions):
        output = enhanced.get("output") if enhanced else None
        debug_payload = dict(enhanced.get("debug", {})) if enhanced else {}
        if output and output.get("description"):
            used = True
            text = str(output["description"]).strip()
            segments.append(
                replace(
                    segment,
                    description=text,
                    enhanced_description=text,
                    debug_available=segment.debug_available or bool(debug_payload),
                    debug={**segment.debug, **debug_payload} if debug_payload else dict(segment.debug),
                )
            )
        else:
            segments.append(
                replace(
                    segment,
                    debug_available=segment.debug_available or bool(debug_payload),
                    debug={**segment.debug, **debug_payload} if debug_payload else dict(segment.debug),
                )
            )
        debug_available = debug_available or bool(debug_payload)

    summary = dict(result.debug_summary)
    summary["text_enhancement"] = {
        "model": text_model,
        "used": used,
        "segment_count": len(result.segments),
    }
    return replace(
        result,
        segments=segments,
        result_source="video_llm+llm" if used and result.video_enhancement_used else ("llm" if used else result.result_source),
        text_model=text_model,
        text_enhancement_used=used,
        debug_available=debug_available,
        debug_summary=summary,
    )


def mark_video_fallback(result: AnalysisResult, reason: str, video_model: str | None = None) -> AnalysisResult:
    summary = dict(result.debug_summary)
    summary["video_enhancement"] = {
        "model": video_model,
        "used": False,
        "skip_reason": reason,
        "segment_count": len(result.segments),
    }
    summary["video_result_stats"] = {
        "success": 0,
        "weak_success": 0,
        "fallback": len(result.segments),
    }
    return replace(
        result,
        segments=[
            replace(
                segment,
                fallback_reason=reason,
                video_result_status="fallback",
                debug_available=segment.debug_available or bool(segment.debug),
            )
            for segment in result.segments
        ],
        video_model=video_model,
        debug_summary=summary,
    )


def mark_text_fallback(result: AnalysisResult, reason: str, text_model: str | None = None) -> AnalysisResult:
    summary = dict(result.debug_summary)
    summary["text_enhancement"] = {
        "model": text_model,
        "used": False,
        "skip_reason": reason,
        "segment_count": len(result.segments),
    }
    return replace(result, text_model=text_model, debug_summary=summary)
