from services.chunking import ChunkPlanner


def test_chunk_planner_creates_overlap():
    planner = ChunkPlanner(chunk_duration_seconds=120, overlap_seconds=2)
    chunks = planner.plan(601)

    assert len(chunks) == 6
    assert chunks[0].start_time == 0.0
    assert chunks[1].start_time == 118.0
    assert chunks[1].overlap_seconds == 2.0
    assert chunks[-1].end_time == 601
