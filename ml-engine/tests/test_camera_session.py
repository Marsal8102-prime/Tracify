import pytest
import numpy as np
import threading
import time

from camera.session import CameraSession, CameraSessionState
from camera.events import CameraEvent
from tests.fakes import FakeCamera, FakePreprocessor, FakeDetector, FakeAligner, FakeEmbedder, FakeRecognizer

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
    session = CameraSession(camera=camera, **fake_components)

    session.start()
    
    # Wait for the thread to process frames and end naturally when camera hits max_reads
    time.sleep(0.5)
    
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
    time.sleep(0.5)
    
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
    time.sleep(0.5)
    
    assert session.frames_processed == 1
    assert session.events_buffered == 0
    session.stop()

def test_event_buffer_append_and_drain(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, **fake_components)
    
    session.start()
    time.sleep(0.5)
    
    assert session.events_buffered == 5
    events = session.get_events()
    assert len(events) == 5
    assert session.events_buffered == 0
    session.stop()

def test_event_buffer_bounded(fake_components):
    camera = FakeCamera(max_reads=10)
    session = CameraSession(camera=camera, buffer_size=3, **fake_components)
    
    session.start()
    time.sleep(0.5)
    
    assert session.events_buffered == 3 # Should be capped at 3
    assert session.status()["events_dropped"] == 7
    session.stop()

def test_event_buffer_drain_with_limit(fake_components):
    camera = FakeCamera(max_reads=5)
    session = CameraSession(camera=camera, **fake_components)
    
    session.start()
    time.sleep(0.5)
    
    events = session.get_events(limit=2)
    assert len(events) == 2
    assert session.events_buffered == 3
    session.stop()

def test_event_fields(fake_components):
    camera = FakeCamera(max_reads=1)
    session = CameraSession(camera=camera, **fake_components)
    
    session.start()
    time.sleep(0.5)
    
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
    
    time.sleep(0.5)
    
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
    session = CameraSession(camera=camera, **fake_components)
    
    session.start()
    time.sleep(0.5)
    session.stop()
    
    status = session.status()
    assert status["state"] == "stopped"
    assert status["frames_processed"] == 5
    assert status["events_produced"] == 5
