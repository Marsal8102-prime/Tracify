from dataclasses import dataclass
from collections import deque
from typing import List, Optional

@dataclass(frozen=True)
class CameraEvent:
    """A single face recognition result from continuous camera processing."""
    person_id: Optional[str]
    recognition_status: str          # "known" or "unknown"
    similarity: float
    threshold: float
    detection_confidence: float
    bbox: List[int]                  # [x1, y1, x2, y2]
    matched_embedding_id: Optional[str]
    timestamp: str                   # ISO-8601 UTC

class EventBuffer:
    """Thread-safe bounded FIFO buffer for CameraEvents."""

    def __init__(self, maxlen: int = 1000):
        self._buffer: deque[CameraEvent] = deque(maxlen=maxlen)
        self._total_produced: int = 0
        self._total_dropped: int = 0

    def append(self, event: CameraEvent) -> None:
        if len(self._buffer) == self._buffer.maxlen:
            self._total_dropped += 1
        self._buffer.append(event)
        self._total_produced += 1

    def drain(self, limit: int = 100) -> List[CameraEvent]:
        result = []
        for _ in range(min(limit, len(self._buffer))):
            try:
                result.append(self._buffer.popleft())
            except IndexError:
                break
        return result

    @property
    def total_produced(self) -> int:
        return self._total_produced

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    def __len__(self) -> int:
        return len(self._buffer)
