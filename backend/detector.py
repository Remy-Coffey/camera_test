from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for environments without scipy
    linear_sum_assignment = None

from domain import BoundingBox, DetectionFrame


@dataclass(slots=True)
class ChunkDetectionResult:
    detections: list[DetectionFrame]
    video_info: dict[str, float | int]
    track_summary: list[dict[str, Any]] = field(default_factory=list)
    sampling_profile: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackState:
    track_id: int
    box: BoundingBox
    first_seen: float
    last_seen: float
    hits: int = 1
    lost_frames: int = 0
    confirmed: bool = False
    enter_events: int = 1
    exit_events: int = 0


_DETECTOR_CACHE: dict[str, YOLO] = {}


def get_model(model_path: str) -> YOLO:
    if model_path not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[model_path] = YOLO(model_path)
    return _DETECTOR_CACHE[model_path]


def _greedy_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    taken_rows: set[int] = set()
    taken_cols: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for row in range(cost_matrix.shape[0]):
        for col in range(cost_matrix.shape[1]):
            pairs.append((row, col, float(cost_matrix[row, col])))
    pairs.sort(key=lambda item: item[2])
    rows: list[int] = []
    cols: list[int] = []
    for row, col, _ in pairs:
        if row in taken_rows or col in taken_cols:
            continue
        taken_rows.add(row)
        taken_cols.add(col)
        rows.append(row)
        cols.append(col)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


class PersonDetector:
    def __init__(self, model_path: Path | str, confidence_threshold: float = 0.4):
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0
        self.scene_change_threshold = 0.22
        self.strong_scene_change_threshold = 0.36
        self.min_confirm_hits = 2
        self.max_lost_frames = 3

    def _frame_signature(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _low_res_edges(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 64, 128)

    def _scene_change_score(
        self,
        previous_signature: np.ndarray | None,
        current_signature: np.ndarray,
        previous_edges: np.ndarray | None,
        current_edges: np.ndarray,
    ) -> float:
        if previous_signature is None or previous_edges is None:
            return 0.0
        hist_score = float(cv2.compareHist(previous_signature.astype("float32"), current_signature.astype("float32"), cv2.HISTCMP_BHATTACHARYYA))
        edge_score = float(np.mean(cv2.absdiff(previous_edges, current_edges)) / 255.0)
        return max(0.0, min(1.0, hist_score * 0.65 + edge_score * 0.35))

    def _bbox_iou(self, left: BoundingBox, right: BoundingBox) -> float:
        ix1 = max(left.x1, right.x1)
        iy1 = max(left.y1, right.y1)
        ix2 = min(left.x2, right.x2)
        iy2 = min(left.y2, right.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        union = left.area + right.area - intersection
        return intersection / union if union > 0 else 0.0

    def _center_distance(self, left: BoundingBox, right: BoundingBox) -> float:
        lx, ly = left.center
        rx, ry = right.center
        return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5

    def _assignment_cost(
        self,
        track: TrackState,
        detection: BoundingBox,
        frame_width: int,
        frame_height: int,
    ) -> float:
        iou = self._bbox_iou(track.box, detection)
        center_distance = self._center_distance(track.box, detection) / max(max(frame_width, frame_height), 1.0)
        scale_delta = abs(track.box.area - detection.area) / max(track.box.area, detection.area, 1.0)
        return (1.0 - iou) * 0.6 + center_distance * 0.3 + scale_delta * 0.1

    def _match_tracks(
        self,
        detections: list[BoundingBox],
        tracks: dict[int, TrackState],
        frame_width: int,
        frame_height: int,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not detections or not tracks:
            return [], set(tracks.keys()), set(range(len(detections)))

        track_ids = list(tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=float)
        for row, track_id in enumerate(track_ids):
            for col, detection in enumerate(detections):
                cost_matrix[row, col] = self._assignment_cost(tracks[track_id], detection, frame_width, frame_height)

        matcher = linear_sum_assignment if linear_sum_assignment is not None else _greedy_assignment
        row_indices, col_indices = matcher(cost_matrix)
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(len(detections)))
        max_valid_cost = 0.82
        for row, col in zip(row_indices.tolist(), col_indices.tolist()):
            track_id = track_ids[row]
            if float(cost_matrix[row, col]) > max_valid_cost:
                continue
            matches.append((track_id, col))
            unmatched_tracks.discard(track_id)
            unmatched_detections.discard(col)
        return matches, unmatched_tracks, unmatched_detections

    def detect_frame(self, model: YOLO, frame: np.ndarray) -> list[BoundingBox]:
        results = model(frame, verbose=False, classes=[self.person_class_id])
        persons: list[BoundingBox] = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                persons.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))
        return persons

    def _detect_at(self, cap: cv2.VideoCapture, model: YOLO, timestamp: float, fps: float) -> tuple[DetectionFrame | None, np.ndarray | None, np.ndarray | None]:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            return None, None, None
        frame_index = int(timestamp * fps) if fps > 0 else 0
        persons = self.detect_frame(model, frame)
        signature = self._frame_signature(frame)
        edges = self._low_res_edges(frame)
        return (
            DetectionFrame(
                frame_index=frame_index,
                timestamp=round(timestamp, 3),
                persons=persons,
                person_count=len(persons),
            ),
            signature,
            edges,
        )

    def process_range(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_interval: float,
        *,
        stable_person_interval: float | None = None,
        refined_sample_interval: float | None = None,
        strong_refined_sample_interval: float | None = None,
    ) -> ChunkDetectionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        model = get_model(self.model_path)
        limit = min(end_time, duration)

        base_interval = max(sample_interval, 0.1)
        stable_interval = max(stable_person_interval or max(sample_interval * 0.75, 0.25), 0.1)
        refined_interval = max(refined_sample_interval or 0.25, 0.1)
        strong_interval = max(strong_refined_sample_interval or 0.1, 0.05)

        active_tracks: dict[int, TrackState] = {}
        track_events: dict[int, dict[str, Any]] = {}
        next_track_id = 1
        previous_signature: np.ndarray | None = None
        previous_edges: np.ndarray | None = None
        previous_frame: DetectionFrame | None = None
        detections_by_ts: dict[float, DetectionFrame] = {}
        sampling_events: list[dict[str, Any]] = []

        def update_tracks(frame: DetectionFrame) -> DetectionFrame:
            nonlocal next_track_id
            matches, unmatched_tracks, unmatched_detections = self._match_tracks(frame.persons, active_tracks, width, height)
            seen_track_ids: list[int] = []

            for track_id, det_index in matches:
                detection = frame.persons[det_index]
                track = active_tracks[track_id]
                detection.track_id = track_id
                track.box = detection
                track.last_seen = frame.timestamp
                track.hits += 1
                track.lost_frames = 0
                track.confirmed = track.confirmed or track.hits >= self.min_confirm_hits
                seen_track_ids.append(track_id)

            for det_index in unmatched_detections:
                detection = frame.persons[det_index]
                track_id = next_track_id
                next_track_id += 1
                detection.track_id = track_id
                active_tracks[track_id] = TrackState(
                    track_id=track_id,
                    box=detection,
                    first_seen=frame.timestamp,
                    last_seen=frame.timestamp,
                    confirmed=self.min_confirm_hits <= 1,
                )
                seen_track_ids.append(track_id)

            removed_track_ids: list[int] = []
            for track_id in unmatched_tracks:
                track = active_tracks[track_id]
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    track.exit_events += 1
                    removed_track_ids.append(track_id)

            for track_id in removed_track_ids:
                active_tracks.pop(track_id, None)

            confirmed_ids: list[int] = []
            for person in frame.persons:
                if person.track_id is None:
                    continue
                track = active_tracks.get(person.track_id)
                if not track:
                    continue
                if track.confirmed:
                    confirmed_ids.append(person.track_id)
                summary = track_events.setdefault(
                    person.track_id,
                    {
                        "track_id": person.track_id,
                        "first_seen": frame.timestamp,
                        "last_seen": frame.timestamp,
                        "hit_count": 0,
                        "lost_count": 0,
                        "enter_events": track.enter_events,
                        "exit_events": 0,
                    },
                )
                summary["last_seen"] = frame.timestamp
                summary["hit_count"] += 1
                summary["enter_events"] = track.enter_events

            for track_id in removed_track_ids:
                summary = track_events.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "first_seen": frame.timestamp,
                        "last_seen": frame.timestamp,
                        "hit_count": 0,
                        "lost_count": 0,
                        "enter_events": 0,
                        "exit_events": 0,
                    },
                )
                summary["lost_count"] += 1
                summary["exit_events"] += 1

            frame.track_ids = sorted(confirmed_ids)
            frame.person_count = len(frame.persons)
            return frame

        def store_frame(frame: DetectionFrame, signature: np.ndarray | None, edges: np.ndarray | None, mode: str) -> DetectionFrame:
            nonlocal previous_signature, previous_edges, previous_frame
            frame.sampling_mode = mode
            frame = update_tracks(frame)
            frame.scene_change_score = round(
                self._scene_change_score(previous_signature, signature, previous_edges, edges) if signature is not None and edges is not None else 0.0,
                4,
            )
            frame.scene_changed = frame.scene_change_score >= self.scene_change_threshold
            detections_by_ts[frame.timestamp] = frame
            previous_signature = signature
            previous_edges = edges
            previous_frame = frame
            return frame

        timestamp = max(0.0, start_time)
        while timestamp <= limit:
            coarse_frame, signature, edges = self._detect_at(cap, model, timestamp, fps)
            if not coarse_frame:
                break

            if previous_frame is not None:
                count_changed = coarse_frame.person_count != previous_frame.person_count
                track_overlap = (
                    len(set(previous_frame.track_ids) & set(coarse_frame.track_ids)) / max(len(set(previous_frame.track_ids) | set(coarse_frame.track_ids)), 1)
                    if previous_frame.track_ids or coarse_frame.track_ids
                    else 1.0
                )
                provisional_scene_score = round(
                    self._scene_change_score(previous_signature, signature, previous_edges, edges) if signature is not None and edges is not None else 0.0,
                    4,
                )
                should_refine = count_changed or provisional_scene_score >= self.scene_change_threshold or track_overlap < 0.35
                if should_refine:
                    event_type = "scene_change" if provisional_scene_score >= self.scene_change_threshold else ("count_change" if count_changed else "track_change")
                    refine_step = strong_interval if provisional_scene_score >= self.strong_scene_change_threshold else refined_interval
                    probe = previous_frame.timestamp + refine_step
                    while probe < coarse_frame.timestamp:
                        refined_frame, refined_signature, refined_edges = self._detect_at(cap, model, probe, fps)
                        if refined_frame:
                            stored = store_frame(refined_frame, refined_signature, refined_edges, "refined")
                            sampling_events.append({"timestamp": stored.timestamp, "event": event_type, "mode": "refined"})
                        probe += refine_step

            interval_mode = "base"
            if coarse_frame.person_count > 0 and stable_interval < base_interval:
                interval_mode = "stable_person"
            stored = store_frame(coarse_frame, signature, edges, interval_mode)
            next_step = stable_interval if stored.person_count > 0 else base_interval
            timestamp = round(timestamp + next_step, 4)

        cap.release()
        ordered_frames = [detections_by_ts[key] for key in sorted(detections_by_ts.keys())]
        for track_id, track in active_tracks.items():
            summary = track_events.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "first_seen": track.first_seen,
                    "last_seen": track.last_seen,
                    "hit_count": track.hits,
                    "lost_count": track.lost_frames,
                    "enter_events": track.enter_events,
                    "exit_events": track.exit_events,
                },
            )
            summary["last_seen"] = track.last_seen
            summary["hit_count"] = max(summary["hit_count"], track.hits)
            summary["lost_count"] = max(summary["lost_count"], track.lost_frames)

        return ChunkDetectionResult(
            detections=ordered_frames,
            video_info={
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": duration,
            },
            track_summary=sorted(track_events.values(), key=lambda item: item["track_id"]),
            sampling_profile={
                "base_sample_interval": base_interval,
                "stable_person_interval": stable_interval,
                "refined_sample_interval": refined_interval,
                "strong_refined_sample_interval": strong_interval,
                "sampling_events": sampling_events,
            },
        )
