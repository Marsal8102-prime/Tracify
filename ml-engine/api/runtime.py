import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from alignment.base import BaseAligner
from alignment.face_aligner import FaceAligner
from config.settings import load_settings, Settings
from detection.base import BaseDetector
from detection.factory import create_detector
from embedding.base import BaseEmbedder
from embedding.arcface_embedder import ArcFaceEmbedder
from preprocessing.preprocessor import FramePreprocessor
from recognition.base import BaseRecognizer
from recognition.embedding_recognizer import EmbeddingRecognizer
from registration.service import RegistrationService
from storage.base import EmbeddingStore
from storage.local_store import LocalEmbeddingStore

logger = logging.getLogger("tracify.api.runtime")

@dataclass
class MLRuntime:
    settings: Optional[Settings]
    preprocessor: Optional[FramePreprocessor]
    detector: Optional[BaseDetector]
    aligner: Optional[BaseAligner]
    embedder: Optional[BaseEmbedder]
    store: Optional[EmbeddingStore]
    recognizer: Optional[BaseRecognizer]
    registration_service: Optional[RegistrationService]
    lock: asyncio.Lock
    ready: bool = False
    error: Optional[str] = None

def initialize_runtime(settings: Optional[Settings] = None, lock: Optional[asyncio.Lock] = None) -> MLRuntime:
    """
    Initializes the ML engine components.
    """
    if lock is None:
        lock = asyncio.Lock()

    try:
        if settings is None:
            settings = load_settings()

        logger.info("Initializing ML runtime components...")

        # 1. Preprocessor
        preprocessor = FramePreprocessor(settings.preprocessing)

        # 2. Detector
        detector = create_detector(settings.detection)
        detector.load_model()

        # 3. Aligner
        aligner = FaceAligner(settings.alignment)

        # 4. Embedder
        embedder = ArcFaceEmbedder(settings.embedding)
        embedder.load_model()

        # 5. Storage
        store = LocalEmbeddingStore(
            storage_dir=settings.storage.embeddings_dir,
            expected_dimension=settings.embedding.dimension
        )

        # 6. Recognizer
        recognizer = EmbeddingRecognizer(
            store=store,
            config=settings.recognition,
            expected_dimension=settings.embedding.dimension
        )
        recognizer.load_gallery()

        # 7. Registration Service
        registration_service = RegistrationService(
            detector=detector,
            aligner=aligner,
            embedder=embedder,
            store=store,
            recognizer=recognizer,
            config=settings.registration,
            embedding_dim=settings.embedding.dimension
        )

        logger.info("ML runtime initialized successfully.")

        return MLRuntime(
            settings=settings,
            preprocessor=preprocessor,
            detector=detector,
            aligner=aligner,
            embedder=embedder,
            store=store,
            recognizer=recognizer,
            registration_service=registration_service,
            lock=lock,
            ready=True
        )
    except Exception:
        logger.error("Failed to initialize ML runtime")
        # Build an empty/dummy runtime with ready=False
        return MLRuntime(
            settings=settings,
            preprocessor=None,
            detector=None,
            aligner=None,
            embedder=None,
            store=None,
            recognizer=None,
            registration_service=None,
            lock=lock,
            ready=False,
            error="Failed to initialize ML models."
        )
