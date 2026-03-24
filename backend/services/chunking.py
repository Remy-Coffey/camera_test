from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(slots=True)
class ChunkSpec:
    index: int
    start_time: float
    end_time: float
    overlap_seconds: float


class ChunkPlanner:
    def __init__(self, chunk_duration_seconds: int, overlap_seconds: int):
        self.chunk_duration_seconds = chunk_duration_seconds
        self.overlap_seconds = overlap_seconds

    def plan(self, duration_seconds: float) -> list[ChunkSpec]:
        if duration_seconds <= 0:
            return [ChunkSpec(index=0, start_time=0.0, end_time=0.0, overlap_seconds=0.0)]

        chunk_count = max(1, ceil(duration_seconds / self.chunk_duration_seconds))
        specs: list[ChunkSpec] = []
        for index in range(chunk_count):
            start = max(0.0, index * self.chunk_duration_seconds - (self.overlap_seconds if index else 0))
            end = min(duration_seconds, (index + 1) * self.chunk_duration_seconds + self.overlap_seconds)
            specs.append(
                ChunkSpec(
                    index=index,
                    start_time=float(start),
                    end_time=float(end),
                    overlap_seconds=float(self.overlap_seconds if index else 0),
                )
            )
        return specs
