from domain import BoundingBox, DetectionFrame
from services.analysis import build_analysis_result


def test_build_analysis_result_merges_overlapping_frames():
    detections = [
        DetectionFrame(frame_index=0, timestamp=0.0, persons=[BoundingBox(0, 0, 10, 10, 0.9)]),
        DetectionFrame(frame_index=1, timestamp=1.0, persons=[BoundingBox(30, 0, 40, 10, 0.9)]),
        DetectionFrame(frame_index=2, timestamp=4.5, persons=[BoundingBox(20, 0, 30, 10, 0.9)]),
    ]

    result = build_analysis_result(
        detections,
        video_duration=20,
        fps=5,
        width=100,
        height=100,
        gap_threshold=2.0,
    )

    assert result.total_segments == 2
    assert result.segments[0].features["movement"] == "moving"
    assert result.segments[1].start_time == 4.5


def test_build_analysis_result_splits_on_person_count_change():
    detections = [
        DetectionFrame(
            frame_index=0,
            timestamp=0.0,
            persons=[BoundingBox(0, 0, 10, 10, 0.9, track_id=1)],
            person_count=1,
            track_ids=[1],
        ),
        DetectionFrame(
            frame_index=1,
            timestamp=1.0,
            persons=[BoundingBox(2, 0, 12, 10, 0.9, track_id=1)],
            person_count=1,
            track_ids=[1],
        ),
        DetectionFrame(
            frame_index=2,
            timestamp=2.0,
            persons=[
                BoundingBox(2, 0, 12, 10, 0.9, track_id=1),
                BoundingBox(40, 0, 50, 10, 0.9, track_id=2),
            ],
            person_count=2,
            track_ids=[1, 2],
        ),
    ]

    result = build_analysis_result(
        detections,
        video_duration=20,
        fps=5,
        width=100,
        height=100,
        gap_threshold=3.0,
    )

    assert result.total_segments == 2
    assert result.segments[0].person_count_range == [1, 1]
    assert result.segments[1].person_count_range == [2, 2]
