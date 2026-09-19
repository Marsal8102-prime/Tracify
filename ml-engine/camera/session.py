import threading
import time
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np

from camera.base import BaseCamera
from preprocessing.preprocessor import FramePreprocessor
from detection.base import BaseDetector
from alignment.base import BaseAligner
from embedding.base import BaseEmbedder
from recognition.base import BaseRecognizer
from camera.events import CameraEvent, EventBuffer

logger = logging.getLogger("tracify.camera.session")

class CameraSessionState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class CameraSession:
    """Manages the camera processing loop."""

    def __init__(
        self,
        camera: BaseCamera,
        preprocessor: FramePreprocessor,
        detector: BaseDetector,
        aligner: BaseAligner,
        embedder: BaseEmbedder,
        recognizer: BaseRecognizer,
        pipeline_lock: threading.Lock,
        buffer_size: int = 1000,
        target_fps: float = 15.0,
        event_cooldown_seconds: float = 5.0,
    ) -> None:
        self._camera = camera
        self._preprocessor = preprocessor
        self._detector = detector
        self._aligner = aligner
        self._embedder = embedder
        self._recognizer = recognizer
        self._pipeline_lock = pipeline_lock
        self._target_fps = target_fps
        self._event_cooldown_seconds = event_cooldown_seconds
        
        self._event_buffer = EventBuffer(maxlen=buffer_size)
        self._state = CameraSessionState.STOPPED
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        self._frames_processed = 0
        self._consecutive_errors = 0
        self._last_error: Optional[str] = None
        self._last_event_times: dict[str, float] = {}

    @property
    def state(self) -> CameraSessionState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == CameraSessionState.RUNNING

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def events_buffered(self) -> int:
        return len(self._event_buffer)

    def start(self) -> None:
        """Starts the camera and the processing thread."""
        if self._state == CameraSessionState.RUNNING:
            raise RuntimeError("Camera session is already running")
        if self._state in (CameraSessionState.STARTING, CameraSessionState.STOPPING):
            raise RuntimeError("Camera session is transitioning")

        self._state = CameraSessionState.STARTING
        self._stop_event.clear()
        self._last_error = None
        self._frames_processed = 0
        self._consecutive_errors = 0
        self._last_event_times.clear()

        try:
            self._camera.open()
        except Exception as e:
            self._state = CameraSessionState.ERROR
            self._last_error = str(e)
            logger.error("Failed to open camera", exc_info=True)
            raise

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="tracify-camera")
        self._thread.start()
        self._state = CameraSessionState.RUNNING

    def stop(self) -> None:
        """Signals the loop to stop and cleans up resources."""
        if self._state == CameraSessionState.STOPPED:
            return
        if self._state == CameraSessionState.ERROR:
            self._state = CameraSessionState.STOPPED
            return

        self._stop_event.set()
        self._state = CameraSessionState.STOPPING
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        try:
            self._camera.release()
        except Exception:
            logger.warning("Error releasing camera", exc_info=True)
            
        self._state = CameraSessionState.STOPPED

    def get_events(self, limit: int = 100) -> List[CameraEvent]:
        """Drain events from the buffer."""
        return self._event_buffer.drain(limit)

    def status(self) -> Dict[str, Any]:
        """Get the current session status."""
        return {
            "state": self._state.value,
            "frames_processed": self._frames_processed,
            "events_buffered": len(self._event_buffer),
            "events_produced": self._event_buffer.total_produced,
            "events_dropped": self._event_buffer.total_dropped,
            "last_error": self._last_error,
        }

    def _should_emit_event(self, event: CameraEvent, now: float) -> bool:
        """Determine whether a CameraEvent should be emitted to the buffer.

        Known-person events (recognition_status == "known" and person_id is
        not None) are suppressed if the same person_id was emitted within the
        cooldown period. All other events are always emitted.
        """
        if event.person_id is None or event.recognition_status != "known":
            return True

        last_time = self._last_event_times.get(event.person_id)
        if last_time is None or (now - last_time) >= self._event_cooldown_seconds:
            self._last_event_times[event.person_id] = now
            return True

        return False

    def _process_frame(self, frame: np.ndarray) -> List[CameraEvent]:
        """Process a single frame. Returns a list of CameraEvents."""
        events = []

        with self._pipeline_lock:
            processed = self._preprocessor.process(frame)
            detections = self._detector.detect(processed.frame)

            for det in detections:
                try:
                    aligned = self._aligner.align(processed.frame, det)
                    if aligned is None:
                        continue

                    embedding = self._embedder.generate(aligned)
                    rec_result = self._recognizer.recognize(embedding)

                    orig_det = det.scale_to_original(processed.scale_factor)

                    events.append(CameraEvent(
                        person_id=rec_result.person_id,
                        recognition_status=rec_result.status.value,
                        similarity=rec_result.similarity,
                        threshold=rec_result.threshold,
                        detection_confidence=float(det.confidence),
                        bbox=[
                            int(orig_det.bbox[0]),
                            int(orig_det.bbox[1]),
                            int(orig_det.bbox[2]),
                            int(orig_det.bbox[3]),
                        ],
                        matched_embedding_id=rec_result.matched_embedding_id,
                        timestamp=rec_result.timestamp,
                    ))
                except Exception:
                    logger.warning("Error processing single face", exc_info=True)
                    continue

        return events

    def _run_loop(self) -> None:
        """Background thread main loop."""
        logger.info("Camera session loop started")

        frame_interval = 1.0 / self._target_fps
        next_process_time = time.monotonic()

        while not self._stop_event.is_set():
            try:
                frame = self._camera.read_frame()
            except Exception:
                logger.warning("Camera read_frame() error", exc_info=True)
                self._stop_event.wait(0.1)
                continue

            if frame is None:
                if not self._camera.is_opened():
                    logger.warning("Camera is no longer opened, stopping session")
                    break
                self._stop_event.wait(0.01)
                continue

            now = time.monotonic()
            if now < next_process_time:
                # Skip intermediate frames
                continue

            next_process_time = max(next_process_time + frame_interval, now)

            self._frames_processed += 1

            try:
                events = self._process_frame(frame)
                now_emit = time.monotonic()
                for event in events:
                    if self._should_emit_event(event, now_emit):
                        self._event_buffer.append(event)
            except Exception:
                logger.warning("Frame processing error", exc_info=True)
                self._consecutive_errors += 1
                if self._consecutive_errors > 100:
                    logger.error("Too many consecutive processing errors, stopping")
                    self._last_error = "Too many consecutive processing errors"
                    break
                continue

            self._consecutive_errors = 0

        self._state = CameraSessionState.STOPPED
        logger.info(
            f"Camera session loop ended: "
            f"frames_processed={self._frames_processed}, "
            f"events_produced={self._event_buffer.total_produced}"
        )
