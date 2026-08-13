"""
Tests — Face registration module (Phase 4).

Covers:
  - Person identity validation (valid, empty, duplicate IDs)
  - Face input (zero, one, multiple faces)
  - Quality validation (image, detection, alignment, embedding)
  - Multiple samples (minimum, maximum, mixed)
  - Duplicate detection (no dup, likely dup, threshold boundary)
  - Storage (success, failure, rollback)
  - Result structure (SUCCESS, REJECTED, DUPLICATE, INVALID)

All tests use synthetic data and mocks for ML components.
No model download or GPU required.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alignment.base import BaseAligner
from config.settings import RecognitionSettings, RegistrationSettings
from detection.base import BaseDetector, DetectionResult
from embedding.base import BaseEmbedder
from recognition import EmbeddingRecognizer
from registration.duplicate import DuplicateChecker, DuplicateCheckResult
from registration.models import PersonIdentity
from registration.quality import FaceQualityValidator
from registration.result import (
    RegistrationResult,
    RegistrationStatus,
    SampleResult,
)
from registration.service import RegistrationService, _generate_embedding_key
from storage import EmbeddingRecord, LocalEmbeddingStore


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_embedding(dim: int = 512, seed: int = 42) -> np.ndarray:
    """Create a deterministic L2-normalized embedding."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(dim).astype(np.float32)
    return raw / np.linalg.norm(raw)


def _make_face_image(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Create a synthetic BGR face image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_aligned_face(seed: int = 0) -> np.ndarray:
    """Create a synthetic 112x112 aligned face."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (112, 112, 3), dtype=np.uint8)


def _make_detection(
    confidence: float = 0.95,
    face_size: float = 150.0,
) -> DetectionResult:
    """Create a valid DetectionResult with 5-point landmarks."""
    return DetectionResult(
        bbox=np.array([100, 80, 100 + face_size, 80 + face_size],
                      dtype=np.float32),
        confidence=confidence,
        landmarks=np.array(
            [
                [140.0, 140.0],
                [210.0, 140.0],
                [175.0, 180.0],
                [145.0, 220.0],
                [205.0, 220.0],
            ],
            dtype=np.float32,
        ),
    )


# ── Mock components ────────────────────────────────────────────────────

class MockDetector(BaseDetector):
    """Mock detector that returns configurable detections."""

    def __init__(self, detections: Optional[List[List[DetectionResult]]] = None):
        """
        Args:
            detections: A list of detection lists, one per call to detect().
                        If None, returns one valid detection per call.
        """
        self._detections = detections
        self._call_count = 0
        self._loaded = True

    def load_model(self) -> None:
        self._loaded = True

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if self._detections is not None:
            if self._call_count < len(self._detections):
                result = self._detections[self._call_count]
                self._call_count += 1
                return result
            return []
        self._call_count += 1
        return [_make_detection()]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class MockAligner(BaseAligner):
    """Mock aligner that returns a synthetic aligned face."""

    def align(self, frame, detection) -> Optional[np.ndarray]:
        if detection.landmarks is None:
            return None
        return _make_aligned_face(seed=int(detection.confidence * 100))

    @property
    def output_size(self):
        return (112, 112)


class MockEmbedder(BaseEmbedder):
    """Mock embedder that returns deterministic normalized embeddings."""

    def __init__(self, dimension: int = 512):
        self._dimension = dimension
        self._loaded = True

    def load_model(self) -> None:
        self._loaded = True

    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        rng = np.random.RandomState(int(aligned_face.sum()) % (2**31))
        raw = rng.randn(self._dimension).astype(np.float32)
        return raw / np.linalg.norm(raw)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimension(self) -> int:
        return self._dimension


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path) -> LocalEmbeddingStore:
    return LocalEmbeddingStore(
        storage_dir=str(tmp_path / "embeddings"),
        expected_dimension=512,
    )


@pytest.fixture
def recognizer(store) -> EmbeddingRecognizer:
    config = RecognitionSettings(
        strategy="cosine",
        similarity_threshold=0.6,
        top_k=3,
    )
    rec = EmbeddingRecognizer(
        store=store,
        config=config,
        expected_dimension=512,
    )
    rec.load_gallery()
    return rec


@pytest.fixture
def reg_config() -> RegistrationSettings:
    return RegistrationSettings(
        minimum_samples=3,
        maximum_samples=10,
        minimum_face_size=80,
        duplicate_threshold=0.7,
        quality_checks_enabled=True,
    )


@pytest.fixture
def service(store, recognizer, reg_config) -> RegistrationService:
    return RegistrationService(
        detector=MockDetector(),
        aligner=MockAligner(),
        embedder=MockEmbedder(),
        store=store,
        recognizer=recognizer,
        config=reg_config,
        embedding_dim=512,
    )


@pytest.fixture
def face_images() -> List[np.ndarray]:
    """5 synthetic face images with different pixel content."""
    return [_make_face_image(seed=i) for i in range(5)]


# ── Test: PersonIdentity model ──────────────────────────────────────────

class TestPersonIdentity:
    """Tests for the PersonIdentity dataclass."""

    def test_create_identity(self):
        """Should create a valid identity with all fields."""
        identity = PersonIdentity(
            person_id="EMP-001",
            display_name="Alice Chen",
            model_version="arcface_buffalo_l",
        )
        assert identity.person_id == "EMP-001"
        assert identity.display_name == "Alice Chen"
        assert identity.status == "active"
        assert identity.embedding_count == 0
        assert identity.registered_at is not None

    def test_identity_metadata(self):
        """Should support arbitrary metadata."""
        identity = PersonIdentity(
            person_id="EMP-002",
            display_name="Bob",
            metadata={"department": "engineering", "role": "SDE"},
        )
        assert identity.metadata["department"] == "engineering"


# ── Test: RegistrationResult ────────────────────────────────────────────

class TestRegistrationResult:
    """Tests for structured registration results."""

    def test_success_result(self):
        """SUCCESS result should have correct fields."""
        result = RegistrationResult(
            person_id="test",
            status=RegistrationStatus.SUCCESS,
            accepted_count=5,
        )
        assert result.is_success
        assert not result.is_duplicate
        assert result.accepted_count == 5
        assert result.timestamp is not None

    def test_rejected_result(self):
        """REJECTED result should include reasons."""
        result = RegistrationResult(
            person_id="test",
            status=RegistrationStatus.REJECTED,
            rejection_reasons=["face too small"],
        )
        assert result.status is RegistrationStatus.REJECTED
        assert "face too small" in result.rejection_reasons

    def test_duplicate_result(self):
        """DUPLICATE result should include candidate info."""
        result = RegistrationResult(
            person_id="new_person",
            status=RegistrationStatus.DUPLICATE,
            duplicate_person_id="existing_person",
            duplicate_similarity=0.85,
        )
        assert result.is_duplicate
        assert result.duplicate_person_id == "existing_person"
        assert result.duplicate_similarity == 0.85

    def test_invalid_result(self):
        """INVALID result should be created correctly."""
        result = RegistrationResult(
            person_id="",
            status=RegistrationStatus.INVALID,
            rejection_reasons=["empty person_id"],
        )
        assert result.status is RegistrationStatus.INVALID
        assert not result.is_success

    def test_status_enum_values(self):
        """Enum values should match expected strings."""
        assert RegistrationStatus.SUCCESS.value == "success"
        assert RegistrationStatus.REJECTED.value == "rejected"
        assert RegistrationStatus.DUPLICATE.value == "duplicate"
        assert RegistrationStatus.INVALID.value == "invalid"


# ── Test: SampleResult ──────────────────────────────────────────────────

class TestSampleResult:
    """Tests for per-sample result tracking."""

    def test_accepted_sample(self):
        result = SampleResult(accepted=True, reason="sample accepted", sample_index=0)
        assert result.accepted
        assert result.sample_index == 0

    def test_rejected_sample(self):
        result = SampleResult(accepted=False, reason="face too small", sample_index=2)
        assert not result.accepted
        assert result.reason == "face too small"


# ── Test: FaceQualityValidator ──────────────────────────────────────────

class TestFaceQualityValidator:
    """Tests for face quality validation."""

    @pytest.fixture
    def validator(self) -> FaceQualityValidator:
        return FaceQualityValidator(
            minimum_face_size=80,
            embedding_dimension=512,
            enabled=True,
        )

    # Image validation
    def test_valid_image(self, validator):
        ok, _ = validator.validate_image(_make_face_image())
        assert ok

    def test_none_image(self, validator):
        ok, reason = validator.validate_image(None)
        assert not ok
        assert "None" in reason

    def test_empty_image(self, validator):
        ok, reason = validator.validate_image(np.array([]))
        assert not ok
        assert "empty" in reason

    def test_wrong_channels_image(self, validator):
        ok, reason = validator.validate_image(
            np.zeros((100, 100), dtype=np.uint8)
        )
        assert not ok
        assert "3-channel" in reason

    # Detection validation
    def test_valid_detection(self, validator):
        ok, _ = validator.validate_detection([_make_detection()])
        assert ok

    def test_no_faces(self, validator):
        ok, reason = validator.validate_detection([])
        assert not ok
        assert "no faces" in reason

    def test_multiple_faces(self, validator):
        ok, reason = validator.validate_detection(
            [_make_detection(), _make_detection()]
        )
        assert not ok
        assert "multiple" in reason

    def test_no_landmarks(self, validator):
        det = DetectionResult(
            bbox=np.array([100, 80, 250, 230], dtype=np.float32),
            confidence=0.9,
            landmarks=None,
        )
        ok, reason = validator.validate_detection([det])
        assert not ok
        assert "no landmarks" in reason

    def test_invalid_landmark_shape(self, validator):
        det = DetectionResult(
            bbox=np.array([100, 80, 250, 230], dtype=np.float32),
            confidence=0.9,
            landmarks=np.zeros((3, 2), dtype=np.float32),
        )
        ok, reason = validator.validate_detection([det])
        assert not ok
        assert "landmark shape" in reason

    def test_nan_landmarks(self, validator):
        det = DetectionResult(
            bbox=np.array([100, 80, 250, 230], dtype=np.float32),
            confidence=0.9,
            landmarks=np.full((5, 2), np.nan, dtype=np.float32),
        )
        ok, reason = validator.validate_detection([det])
        assert not ok
        assert "NaN" in reason

    def test_face_too_small(self, validator):
        det = _make_detection(face_size=50.0)  # below 80px minimum
        ok, reason = validator.validate_detection([det])
        assert not ok
        assert "too small" in reason

    # Aligned face validation
    def test_valid_aligned(self, validator):
        ok, _ = validator.validate_aligned(_make_aligned_face())
        assert ok

    def test_none_aligned(self, validator):
        ok, reason = validator.validate_aligned(None)
        assert not ok
        assert "None" in reason

    def test_empty_aligned(self, validator):
        ok, reason = validator.validate_aligned(np.array([]))
        assert not ok
        assert "empty" in reason

    # Embedding validation
    def test_valid_embedding(self, validator):
        ok, _ = validator.validate_embedding(_make_embedding())
        assert ok

    def test_none_embedding(self, validator):
        ok, reason = validator.validate_embedding(None)
        assert not ok
        assert "None" in reason

    def test_wrong_dim_embedding(self, validator):
        ok, reason = validator.validate_embedding(_make_embedding(dim=256))
        assert not ok
        assert "dimension mismatch" in reason

    def test_nan_embedding(self, validator):
        ok, reason = validator.validate_embedding(
            np.full(512, np.nan, dtype=np.float32)
        )
        assert not ok
        assert "NaN" in reason

    def test_inf_embedding(self, validator):
        ok, reason = validator.validate_embedding(
            np.full(512, np.inf, dtype=np.float32)
        )
        assert not ok
        assert "NaN or Inf" in reason

    def test_zero_embedding(self, validator):
        ok, reason = validator.validate_embedding(
            np.zeros(512, dtype=np.float32)
        )
        assert not ok
        assert "zero vector" in reason

    def test_2d_embedding(self, validator):
        ok, reason = validator.validate_embedding(
            np.random.randn(1, 512).astype(np.float32)
        )
        assert not ok
        assert "1-dimensional" in reason

    # Disabled checks
    def test_disabled_checks_pass(self):
        validator = FaceQualityValidator(enabled=False)
        ok, _ = validator.validate_image(None)
        assert ok
        ok, _ = validator.validate_detection([])
        assert ok
        ok, _ = validator.validate_aligned(None)
        assert ok
        ok, _ = validator.validate_embedding(None)
        assert ok


# ── Test: DuplicateChecker ──────────────────────────────────────────────

class TestDuplicateChecker:
    """Tests for duplicate identity detection."""

    def test_no_duplicate_empty_gallery(self, store, recognizer):
        """Empty gallery should never find duplicates."""
        checker = DuplicateChecker(recognizer)
        result = checker.check(_make_embedding(seed=1), threshold=0.7)
        assert not result.is_duplicate

    def test_no_duplicate_different_person(self, store, recognizer):
        """Different embedding should not be flagged as duplicate."""
        store.save(EmbeddingRecord(
            person_id="existing",
            embedding=_make_embedding(seed=100),
        ))
        recognizer.refresh_gallery()
        checker = DuplicateChecker(recognizer)
        result = checker.check(_make_embedding(seed=9999), threshold=0.7)
        assert not result.is_duplicate

    def test_duplicate_detected(self, store, recognizer):
        """Same embedding should be detected as duplicate."""
        emb = _make_embedding(seed=42)
        store.save(EmbeddingRecord(person_id="existing", embedding=emb))
        recognizer.refresh_gallery()
        checker = DuplicateChecker(recognizer)
        result = checker.check(emb, threshold=0.7)
        assert result.is_duplicate
        assert result.matched_person_id == "existing"
        assert result.similarity > 0.99

    def test_threshold_boundary(self, store, recognizer):
        """Similarity exactly below threshold should not be a duplicate."""
        emb = _make_embedding(seed=42)
        store.save(EmbeddingRecord(person_id="existing", embedding=emb))
        recognizer.refresh_gallery()
        checker = DuplicateChecker(recognizer)
        # With threshold=1.1 (impossible to reach), nothing is duplicate
        result = checker.check(emb, threshold=1.1)
        assert not result.is_duplicate


# ── Test: Embedding key generation ──────────────────────────────────────

class TestEmbeddingKeyGeneration:
    """Tests for composite embedding storage keys."""

    def test_key_format(self):
        key = _generate_embedding_key("alice", 0)
        assert key == "alice__emb_0"

    def test_key_with_index(self):
        key = _generate_embedding_key("bob", 3)
        assert key == "bob__emb_3"


# ── Test: Identity validation ───────────────────────────────────────────

class TestIdentityValidation:
    """Tests for person ID validation during registration."""

    def test_empty_person_id(self, service, face_images):
        result = service.register("", "Name", face_images)
        assert result.status is RegistrationStatus.INVALID

    def test_whitespace_person_id(self, service, face_images):
        result = service.register("   ", "Name", face_images)
        assert result.status is RegistrationStatus.INVALID

    def test_reserved_separator_in_id(self, service, face_images):
        result = service.register("bad__emb_id", "Name", face_images)
        assert result.status is RegistrationStatus.INVALID

    def test_duplicate_person_id(self, service, store, face_images):
        """Registering with an already-used person_id should be invalid."""
        # Pre-store an embedding with this person_id prefix
        store.save(EmbeddingRecord(
            person_id="existing__emb_0",
            embedding=_make_embedding(seed=999),
        ))
        result = service.register("existing", "Name", face_images)
        assert result.status is RegistrationStatus.INVALID

    def test_valid_person_id(self, service, face_images):
        result = service.register("EMP-001", "Alice", face_images)
        # Should not be INVALID (may be SUCCESS or other status)
        assert result.status is not RegistrationStatus.INVALID


# ── Test: Face input scenarios ──────────────────────────────────────────

class TestFaceInput:
    """Tests for different face input scenarios."""

    def test_no_images(self, service):
        result = service.register("EMP-001", "Alice", [])
        assert result.status is RegistrationStatus.REJECTED

    def test_zero_faces_detected(self, store, recognizer, reg_config):
        """All samples with no faces should be rejected."""
        detector = MockDetector(
            detections=[[] for _ in range(5)]  # No faces in any image
        )
        svc = RegistrationService(
            detector=detector,
            aligner=MockAligner(),
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=reg_config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(5)]
        result = svc.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.REJECTED
        assert result.accepted_count == 0

    def test_multiple_faces_detected(self, store, recognizer, reg_config):
        """Samples with multiple faces should be rejected."""
        detector = MockDetector(
            detections=[
                [_make_detection(), _make_detection()]  # Two faces
                for _ in range(5)
            ]
        )
        svc = RegistrationService(
            detector=detector,
            aligner=MockAligner(),
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=reg_config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(5)]
        result = svc.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.REJECTED
        assert result.accepted_count == 0


# ── Test: Multiple samples ──────────────────────────────────────────────

class TestMultipleSamples:
    """Tests for multiple face sample handling."""

    def test_sufficient_valid_samples(self, service, face_images):
        """5 valid samples should pass minimum_samples=3 requirement."""
        result = service.register("EMP-001", "Alice", face_images)
        assert result.status is RegistrationStatus.SUCCESS
        assert result.accepted_count >= 3

    def test_insufficient_valid_samples(self, store, recognizer):
        """Below minimum should be rejected."""
        config = RegistrationSettings(
            minimum_samples=5,
            maximum_samples=10,
            minimum_face_size=80,
            duplicate_threshold=0.7,
        )
        # Only 3 images with valid faces, need 5
        detector = MockDetector(
            detections=[
                [_make_detection()],
                [_make_detection()],
                [_make_detection()],
            ]
        )
        svc = RegistrationService(
            detector=detector,
            aligner=MockAligner(),
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(3)]
        result = svc.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.REJECTED

    def test_maximum_samples_trimmed(self, store, recognizer):
        """Excess samples should be trimmed to maximum."""
        config = RegistrationSettings(
            minimum_samples=1,
            maximum_samples=3,
            minimum_face_size=80,
            duplicate_threshold=0.7,
        )
        svc = RegistrationService(
            detector=MockDetector(),
            aligner=MockAligner(),
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(10)]
        result = svc.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.SUCCESS
        assert result.accepted_count <= 3

    def test_mixed_valid_invalid_samples(self, store, recognizer, reg_config):
        """Mix of valid and invalid samples should still work if enough pass."""
        # 5 images: 3 with valid face, 2 with no face
        detector = MockDetector(
            detections=[
                [_make_detection()],
                [],  # no face
                [_make_detection()],
                [],  # no face
                [_make_detection()],
            ]
        )
        svc = RegistrationService(
            detector=detector,
            aligner=MockAligner(),
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=reg_config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(5)]
        result = svc.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.SUCCESS
        assert result.accepted_count == 3
        assert result.rejected_count == 2

    def test_sample_results_tracked(self, service, face_images):
        """Per-sample results should be returned."""
        result = service.register("EMP-001", "Alice", face_images)
        assert len(result.sample_results) == len(face_images)
        for sr in result.sample_results:
            assert isinstance(sr, SampleResult)
            assert isinstance(sr.sample_index, int)


# ── Test: Duplicate detection (via service) ─────────────────────────────

class TestRegistrationDuplicate:
    """Tests for duplicate detection during registration."""

    def test_no_duplicate_new_person(self, service, face_images):
        """First registration should not be flagged as duplicate."""
        result = service.register("EMP-001", "Alice", face_images)
        assert result.status is RegistrationStatus.SUCCESS

    def test_duplicate_detected_via_service(
        self, store, recognizer, reg_config,
    ):
        """Registering someone whose face matches an existing person."""
        # Store an existing person
        existing_emb = _make_embedding(seed=42)
        store.save(EmbeddingRecord(
            person_id="existing__emb_0",
            embedding=existing_emb,
        ))
        recognizer.refresh_gallery()

        # Create a service where the mock embedder always returns
        # the same embedding as the existing person
        class FixedEmbedder(BaseEmbedder):
            def load_model(self): pass
            def generate(self, face): return existing_emb
            @property
            def is_loaded(self): return True
            @property
            def dimension(self): return 512

        svc = RegistrationService(
            detector=MockDetector(),
            aligner=MockAligner(),
            embedder=FixedEmbedder(),
            store=store,
            recognizer=recognizer,
            config=reg_config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(5)]
        result = svc.register("EMP-NEW", "New Person", images)
        assert result.status is RegistrationStatus.DUPLICATE
        assert result.duplicate_person_id is not None
        assert result.duplicate_similarity is not None
        assert result.duplicate_similarity > 0.7


# ── Test: Storage ───────────────────────────────────────────────────────

class TestRegistrationStorage:
    """Tests for embedding storage during registration."""

    def test_embeddings_stored(self, service, store, face_images):
        """Successful registration should store embeddings."""
        result = service.register("EMP-001", "Alice", face_images)
        assert result.status is RegistrationStatus.SUCCESS
        assert store.count() == result.accepted_count

    def test_storage_uses_composite_keys(self, service, store, face_images):
        """Stored embeddings should use person_id__emb_N keys."""
        result = service.register("EMP-001", "Alice", face_images)
        assert result.status is RegistrationStatus.SUCCESS
        ids = store.list_ids()
        for stored_id in ids:
            assert stored_id.startswith("EMP-001__emb_")

    def test_storage_failure_rollback(self, store, recognizer, reg_config):
        """Storage failure should trigger rollback of any stored embeddings."""
        call_count = 0
        original_save = store.save

        def failing_save(record):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise IOError("Disk full")
            original_save(record)

        store.save = failing_save

        svc = RegistrationService(
            detector=MockDetector(),
            aligner=MockAligner(),
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=reg_config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(5)]
        result = svc.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.INVALID
        assert "storage failure" in result.rejection_reasons[0]

    def test_no_orphans_on_rejection(self, store, recognizer, reg_config):
        """Rejected registration should not leave any embeddings."""
        # All samples have no face → rejected, no storage
        detector = MockDetector(detections=[[] for _ in range(5)])
        svc = RegistrationService(
            detector=detector,
            aligner=MockAligner(),
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=reg_config,
            embedding_dim=512,
        )
        images = [_make_face_image(seed=i) for i in range(5)]
        result = svc.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.REJECTED
        assert store.count() == 0
