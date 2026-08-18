import asyncio
import numpy as np
from typing import List, Optional

from api.runtime import MLRuntime
from alignment.base import BaseAligner
from detection.base import BaseDetector, DetectionResult
from embedding.base import BaseEmbedder
from preprocessing.preprocessor import FramePreprocessor, PreprocessedFrame
from recognition.base import BaseRecognizer
from recognition.result import RecognitionResult, RecognitionStatus
from registration.service import RegistrationService
from registration.result import RegistrationResult, RegistrationStatus, SampleResult
from config.settings import Settings, PreprocessingSettings

class FakePreprocessor(FramePreprocessor):
    def __init__(self):
        super().__init__(PreprocessingSettings())

    def process(self, frame: np.ndarray) -> PreprocessedFrame:
        return PreprocessedFrame(
            frame=frame,
            original_shape=(frame.shape[0], frame.shape[1]),
            scale_factor=1.0
        )

class FakeDetector(BaseDetector):
    def __init__(self, should_detect: bool = True):
        self.should_detect = should_detect
        self._loaded = True

    def load_model(self) -> None:
        pass

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if not self.should_detect:
            return []
        return [
            DetectionResult(
                bbox=np.array([0.0, 0.0, 100.0, 100.0]),
                landmarks=np.zeros((5, 2)),
                confidence=0.99
            )
        ]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

class FakeAligner(BaseAligner):
    def align(self, frame: np.ndarray, detection: DetectionResult) -> Optional[np.ndarray]:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    @property
    def output_size(self):
        return (112, 112)

class FakeEmbedder(BaseEmbedder):
    def __init__(self):
        self._loaded = True

    def load_model(self) -> None:
        pass

    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        return np.ones((512,), dtype=np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimension(self) -> int:
        return 512

class FakeRecognizer(BaseRecognizer):
    def __init__(self):
        self._size = 0

    def recognize(self, query_embedding: np.ndarray) -> RecognitionResult:
        return RecognitionResult(
            status=RecognitionStatus.KNOWN,
            person_id="EMP-001",
            similarity=0.9,
            matched_embedding_id="EMP-001__0",
            threshold=0.6
        )

    def recognize_batch(self, query_embeddings: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize(e) for e in query_embeddings]

    def load_gallery(self) -> int:
        return 1

    def refresh_gallery(self) -> int:
        return 1

    @property
    def gallery_size(self) -> int:
        return self._size

class FakeRegistrationService:
    def register(self, person_id: str, display_name: str, face_images: List[np.ndarray], metadata: Optional[dict] = None) -> RegistrationResult:
        return RegistrationResult(
            person_id=person_id,
            status=RegistrationStatus.SUCCESS,
            accepted_count=len(face_images),
            rejected_count=0,
            rejection_reasons=[],
            sample_results=[
                SampleResult(accepted=True, reason="ok", sample_index=i)
                for i in range(len(face_images))
            ]
        )

def create_fake_runtime(ready: bool = True, lock: Optional[asyncio.Lock] = None) -> MLRuntime:
    if lock is None:
        lock = asyncio.Lock()

    return MLRuntime(
        settings=Settings(),
        preprocessor=FakePreprocessor(),
        detector=FakeDetector(),
        aligner=FakeAligner(),
        embedder=FakeEmbedder(),
        store=None, # type: ignore
        recognizer=FakeRecognizer(),
        registration_service=FakeRegistrationService(), # type: ignore
        lock=lock,
        ready=ready
    )
