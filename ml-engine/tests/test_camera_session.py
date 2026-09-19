import pytest
import numpy as np
import threading
import time

from unittest.mock import patch
import itertools

from camera.session import CameraSession, CameraSessionState
from camera.events import CameraEvent
from recognition.result import RecognitionResult, RecognitionStatus
from tests.fakes import FakeCamera, FakePreprocessor, FakeDetector, FakeAligner, FakeEmbedder, FakeRecognizer

@pytest.fixture(autouse=True)
def mock_monotonic():
    counter = itertools.count(start=0.0, step=0.1)
    with patch("camera.session.time.monotonic", side_effect=counter) as mock_m:
        yield mock_m

@pytest.fixture
def fake_components():
    return {
        "preprocessor": FakePreprocessor(),
        "detector": FakeDetector(),
        "aligner": FakeAligner(),
        "embedder": FakeEmbedder(),
        "recognizer": FakeRecognizer(),
        "pipeline_lock": threading.Lock()
    }

def test_start_and_stop(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, **fake_components)

    assert not session.is_running
    assert session.state == CameraSessionState.STOPPED

    session.start()
    assert session.is_running
    assert session.state == CameraSessionState.RUNNING

    session.stop()
    assert not session.is_running
    assert session.state == CameraSessionState.STOPPED

def test_start_already_running_raises(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, **fake_components)

    session.start()
    with pytest.raises(RuntimeError, match="Camera session is already running"):
        session.start()
    session.stop()

def test_stop_when_stopped_is_idempotent(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, **fake_components)

    session.stop() # Should not raise

def test_restart_after_stop(fake_components):
    camera = FakeCamera(max_reads=2)
    session = CameraSession(camera=camera, **fake_components)

    session.start()
    session.stop()

    session.start()
    assert session.is_running
    session.stop()

def test_camera_open_failure(fake_components):
    class FailingCamera(FakeCamera):
        def open(self):
            raise ValueError("Camera disconnected")
    
    session = CameraSession(camera=FailingCamera(), **fake_components)

    with pytest.raises(ValueError, match="Camera disconnected"):
        session.start()
    assert session.state == CameraSessionState.ERROR

def test_processes_frames_and_produces_events(fake_components):
    camera = FakeCamera(max_reads=3)
    session = CameraSession(camera=camera, event_cooldown_seconds=0.0, **fake_components)

    session.start()
    
    # Wait for the thread to process frames and end naturally when camera hits max_reads
    if session._thread:
        session._thread.join(timeout=5.0)
    
    assert session.frames_processed == 3
    assert session.events_buffered == 3
    session.stop()

def test_no_faces_detected_produces_no_events(fake_components):
    camera = FakeCamera(max_reads=3)
    detector = FakeDetector(should_detect=False)
    comps = dict(fake_components)
    comps["detector"] = detector

    session = CameraSession(camera=camera, **comps)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    
    assert session.frames_processed == 3
    assert session.events_buffered == 0
    session.stop()

def test_alignment_returns_none_skips_face(fake_components):
    camera = FakeCamera(max_reads=1)
    
    class SkipAligner(FakeAligner):
        def align(self, frame, detection):
            return None
    
    comps = dict(fake_components)
    comps["aligner"] = SkipAligner()

    session = CameraSession(camera=camera, **comps)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    
    assert session.frames_processed == 1
    assert session.events_buffered == 0
    session.stop()

def test_event_buffer_append_and_drain(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, event_cooldown_seconds=0.0, **fake_components)
    
    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    
    assert session.events_buffered == 5
    events = session.get_events()
    assert len(events) == 5
    assert session.events_buffered == 0
    session.stop()

def test_event_buffer_bounded(fake_components):
    camera = FakeCamera(max_reads=10)
    session = CameraSession(camera=camera, buffer_size=3, event_cooldown_seconds=0.0, **fake_components)
    
    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    
    assert session.events_buffered == 3 # Should be capped at 3
    assert session.status()["events_dropped"] == 7
    session.stop()

def test_event_buffer_drain_with_limit(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, event_cooldown_seconds=0.0, **fake_components)
    
    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    
    events = session.get_events(limit=2)
    assert len(events) == 2
    assert session.events_buffered == 3
    session.stop()

def test_event_fields(fake_components):
    camera = FakeCamera(max_reads=1)
    session = CameraSession(camera=camera, **fake_components)
    
    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    
    events = session.get_events()
    assert len(events) == 1
    event = events[0]
    
    assert event.person_id == "EMP-001"
    assert event.recognition_status == "known"
    assert event.similarity == 0.9
    assert event.threshold == 0.6
    assert event.detection_confidence == 0.99
    assert len(event.bbox) == 4
    assert event.matched_embedding_id == "EMP-001__0"
    assert hasattr(event, "timestamp")
    session.stop()

def test_consecutive_error_limit(fake_components):
    camera = FakeCamera(max_reads=200)
    
    class FailingDetector(FakeDetector):
        def detect(self, frame):
            raise Exception("Detection failed")

    comps = dict(fake_components)
    comps["detector"] = FailingDetector()

    session = CameraSession(camera=camera, **comps)
    session.start()
    
    if session._thread:
        session._thread.join(timeout=5.0)
    
    assert session.state == CameraSessionState.STOPPED
    assert session.status()["last_error"] == "Too many consecutive processing errors"

def test_stop_during_processing(fake_components):
    camera = FakeCamera(max_reads=1000)
    session = CameraSession(camera=camera, **fake_components)
    
    session.start()
    time.sleep(0.1) # Let it process some
    session.stop()
    assert session.state == CameraSessionState.STOPPED
    assert session.frames_processed > 0

def test_daemon_thread(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, **fake_components)
    session.start()
    assert session._thread is not None
    assert session._thread.daemon is True
    session.stop()

def test_status_while_running(fake_components):
    camera = FakeCamera(max_reads=100)
    session = CameraSession(camera=camera, **fake_components)
    
    session.start()
    status = session.status()
    assert status["state"] == "running"
    session.stop()

def test_status_after_stop(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, event_cooldown_seconds=0.0, **fake_components)
    
    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    session.stop()
    
    status = session.status()
    assert status["state"] == "stopped"
    assert status["frames_processed"] == 5
    assert status["events_produced"] == 5

def test_rate_limiting_skips_frames(fake_components):
    # Mock time advances by 0.1s on each call to time.monotonic().
    # Target FPS is 1.0 (interval = 1.0s).
    # Phase 4D adds one extra monotonic() call per processed frame for
    # cooldown evaluation, shifting subsequent timing by 0.1s per processed
    # frame. With 20 reads the FPS pattern still yields exactly 3 processed
    # frames.  Cooldown is disabled so all processed events are emitted.
    camera = FakeCamera(max_reads=20)
    session = CameraSession(camera=camera, target_fps=1.0, event_cooldown_seconds=0.0, **fake_components)
    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)
    session.stop()
    status = session.status()
    assert status["frames_processed"] == 3
    assert status["events_produced"] == 3


# --- Helpers for Phase 4D cooldown tests ---

class SequentialRecognizer(FakeRecognizer):
    """Returns a pre-defined sequence of RecognitionResults, one per call."""
    def __init__(self, results):
        super().__init__()
        self._results = results
        self._index = 0

    def recognize(self, query_embedding):
        result = self._results[self._index % len(self._results)]
        self._index += 1
        return result


class UnknownRecognizer(FakeRecognizer):
    """Always returns an unknown recognition result."""
    def recognize(self, query_embedding):
        return RecognitionResult(
            status=RecognitionStatus.UNKNOWN,
            person_id=None,
            similarity=0.3,
            matched_embedding_id=None,
            threshold=0.6,
        )


# --- Phase 4D: Cooldown tests ---

def test_cooldown_first_known_event_emitted(fake_components):
    """First recognition of a known person should be emitted."""
    camera = FakeCamera(max_reads=1)
    session = CameraSession(camera=camera, event_cooldown_seconds=5.0, **fake_components)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)

    assert session.frames_processed == 1
    assert session.events_buffered == 1

    events = session.get_events()
    assert events[0].person_id == "EMP-001"
    assert events[0].recognition_status == "known"
    session.stop()


def test_cooldown_same_person_within_cooldown_suppressed(fake_components):
    """Same person_id within cooldown period should be suppressed.

    With mock_monotonic advancing 0.1s per call and 2 monotonic() calls
    per processed frame (FPS check + cooldown check), the cooldown
    timestamps are:
      Frame 1: cooldown_time=0.2 -> emit, record 0.2
      Frame 2: cooldown_time=0.4 -> 0.4 - 0.2 = 0.2 < 5.0 -> suppress
      Frame 3: cooldown_time=0.6 -> 0.6 - 0.2 = 0.4 < 5.0 -> suppress
    """
    camera = FakeCamera(max_reads=3)
    session = CameraSession(camera=camera, event_cooldown_seconds=5.0, **fake_components)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)

    assert session.frames_processed == 3
    assert session.events_buffered == 1
    session.stop()


def test_cooldown_same_person_after_cooldown_emitted(fake_components):
    """Same person_id after cooldown expires should be emitted again.

    With mock_monotonic advancing 0.1s per call and 2 monotonic() calls
    per processed frame (FPS check + cooldown check):
      Frame 1: cooldown_time=0.2 -> emit, record 0.2
      Frame 2: cooldown_time=0.4 -> suppress (0.2 < 1.0)
      Frame 3: cooldown_time=0.6 -> suppress (0.4 < 1.0)
      Frame 4: cooldown_time=0.8 -> suppress (0.6 < 1.0)
      Frame 5: cooldown_time=1.0 -> suppress (0.8 < 1.0)
      Frame 6: cooldown_time=1.2 -> 1.2 - 0.2 = 1.0 >= 1.0 -> emit
    """
    camera = FakeCamera(max_reads=6)
    session = CameraSession(camera=camera, event_cooldown_seconds=1.0, **fake_components)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)

    assert session.frames_processed == 6
    assert session.events_buffered == 2
    session.stop()


def test_cooldown_different_persons_independent(fake_components):
    """Different person_ids should have independent cooldowns.

    Monotonic call trace (0.1s step):
      Frame 1: cooldown_time=0.2 -> person A, first -> emit
      Frame 2: cooldown_time=0.4 -> person B, first -> emit
      Frame 3: cooldown_time=0.6 -> person A, 0.6-0.2=0.4 < 5.0 -> suppress
      Frame 4: cooldown_time=0.8 -> person B, 0.8-0.4=0.4 < 5.0 -> suppress
    """
    result_a = RecognitionResult(
        status=RecognitionStatus.KNOWN,
        person_id="PERSON-A",
        similarity=0.9,
        matched_embedding_id="PERSON-A__0",
        threshold=0.6,
    )
    result_b = RecognitionResult(
        status=RecognitionStatus.KNOWN,
        person_id="PERSON-B",
        similarity=0.85,
        matched_embedding_id="PERSON-B__0",
        threshold=0.6,
    )
    recognizer = SequentialRecognizer([result_a, result_b, result_a, result_b])
    comps = dict(fake_components)
    comps["recognizer"] = recognizer

    camera = FakeCamera(max_reads=4)
    session = CameraSession(camera=camera, event_cooldown_seconds=5.0, **comps)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)

    assert session.frames_processed == 4
    assert session.events_buffered == 2

    events = session.get_events()
    assert events[0].person_id == "PERSON-A"
    assert events[1].person_id == "PERSON-B"
    session.stop()


def test_cooldown_unknown_events_always_emitted(fake_components):
    """Unknown events (person_id=None) should never be suppressed."""
    recognizer = UnknownRecognizer()
    comps = dict(fake_components)
    comps["recognizer"] = recognizer

    camera = FakeCamera(max_reads=3)
    session = CameraSession(camera=camera, event_cooldown_seconds=5.0, **comps)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)

    assert session.frames_processed == 3
    assert session.events_buffered == 3

    events = session.get_events()
    for e in events:
        assert e.person_id is None
        assert e.recognition_status == "unknown"
    session.stop()


def test_cooldown_suppressed_events_not_in_buffer(fake_components):
    """Suppressed events must not appear in EventBuffer at all.

    With 3 frames of the same known person and 5s cooldown, only
    the first event is produced.  The other two are suppressed and
    never enter the buffer.
    """
    camera = FakeCamera(max_reads=3)
    session = CameraSession(camera=camera, event_cooldown_seconds=5.0, **fake_components)

    session.start()
    if session._thread:
        session._thread.join(timeout=5.0)

    status = session.status()
    assert status["events_produced"] == 1
    assert status["events_buffered"] == 1
    assert status["events_dropped"] == 0
    session.stop()
