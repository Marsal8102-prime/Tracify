"""
Tests — End-to-end Phase 4 registration integration test.

Verifies the complete registration-to-recognition chain:
    Face Images → Detection → Alignment → Embedding → Quality Check →
    Duplicate Check → Storage → Gallery Refresh → Recognition

Demonstrates that after registration, the Phase 3 recognition engine
can successfully identify the newly registered person.

All tests use synthetic data and mock components — no model download
or GPU required.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from alignment import FaceAligner
from config.settings import (
    AlignmentSettings,
    RecognitionSettings,
    RegistrationSettings,
)
from detection.base import BaseDetector, DetectionResult
from embedding.base import BaseEmbedder
from recognition import EmbeddingRecognizer, RecognitionStatus
from registration import (
    RegistrationResult,
    RegistrationService,
    RegistrationStatus,
)
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


def _make_detection(confidence: float = 0.95) -> DetectionResult:
    return DetectionResult(
        bbox=np.array([100, 80, 250, 280], dtype=np.float32),
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


# ── Mock components (same as test_registration.py) ─────────────────────

class MockDetector(BaseDetector):
    def __init__(self):
        self._loaded = True

    def load_model(self): self._loaded = True

    def detect(self, frame):
        return [_make_detection()]

    @property
    def is_loaded(self): return self._loaded


class MockEmbedder(BaseEmbedder):
    """
    Mock embedder that produces deterministic embeddings based on
    pixel content. Same image → same embedding (for identity verification).
    """

    def __init__(self, dim: int = 512):
        self._dim = dim
        self._loaded = True

    def load_model(self): self._loaded = True

    def generate(self, aligned_face: np.ndarray) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Not loaded")
        rng = np.random.RandomState(int(aligned_face.mean() * 1000) % (2**31))
        raw = rng.randn(self._dim).astype(np.float32)
        return raw / np.linalg.norm(raw)

    @property
    def is_loaded(self): return self._loaded

    @property
    def dimension(self): return self._dim


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path) -> LocalEmbeddingStore:
    return LocalEmbeddingStore(
        storage_dir=str(tmp_path / "embeddings"),
        expected_dimension=512,
    )


@pytest.fixture
def aligner() -> FaceAligner:
    return FaceAligner(AlignmentSettings(output_size=[112, 112]))


@pytest.fixture
def recognizer(store) -> EmbeddingRecognizer:
    config = RecognitionSettings(
        strategy="cosine",
        similarity_threshold=0.6,
        top_k=3,
    )
    rec = EmbeddingRecognizer(
        store=store, config=config, expected_dimension=512,
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
def service(
    store, aligner, recognizer, reg_config,
) -> RegistrationService:
    return RegistrationService(
        detector=MockDetector(),
        aligner=aligner,
        embedder=MockEmbedder(),
        store=store,
        recognizer=recognizer,
        config=reg_config,
        embedding_dim=512,
    )


# ── End-to-end integration tests ────────────────────────────────────────

class TestPhase4EndToEnd:
    """
    End-to-end tests: Registration → Storage → Recognition.
    """

    def test_register_then_recognize(
        self, service, recognizer, store,
    ):
        """
        Register a person, then verify they can be recognized by Phase 3.
        """
        # Step 1: Register
        images = [_make_face_image(seed=42) for _ in range(5)]
        result = service.register("EMP-001", "Alice Chen", images)

        assert result.status is RegistrationStatus.SUCCESS
        assert result.accepted_count >= 3
        assert result.person_id == "EMP-001"

        # Step 2: Verify storage
        stored_count = store.count()
        assert stored_count == result.accepted_count

        # Step 3: Verify recognition
        # Use the same image → same embedding → should be recognized
        # We need to generate the same embedding the service would produce
        # Since all images are the same (seed=42), all embeddings are identical
        embedder = MockEmbedder()
        aligner = FaceAligner(AlignmentSettings(output_size=[112, 112]))
        detection = _make_detection()
        aligned = aligner.align(images[0], detection)
        query_embedding = embedder.generate(aligned)

        rec_result = recognizer.recognize(query_embedding)
        assert rec_result.status is RecognitionStatus.KNOWN
        assert rec_result.person_id.startswith("EMP-001")

    def test_register_two_people_distinguish(
        self, service, recognizer, store,
    ):
        """
        Register two different people, verify each is correctly identified.
        """
        # Register Alice (same images for consistency)
        alice_images = [_make_face_image(seed=100) for _ in range(5)]
        result_a = service.register("ALICE", "Alice", alice_images)
        assert result_a.status is RegistrationStatus.SUCCESS

        # Register Bob (different images)
        bob_images = [_make_face_image(seed=200) for _ in range(5)]
        result_b = service.register("BOB", "Bob", bob_images)
        assert result_b.status is RegistrationStatus.SUCCESS

        # Verify storage
        assert store.count() == result_a.accepted_count + result_b.accepted_count

        # Verify recognition distinguishes them
        embedder = MockEmbedder()
        aligner = FaceAligner(AlignmentSettings(output_size=[112, 112]))
        detection = _make_detection()

        # Alice query
        alice_aligned = aligner.align(alice_images[0], detection)
        alice_emb = embedder.generate(alice_aligned)
        alice_rec = recognizer.recognize(alice_emb)
        assert alice_rec.status is RecognitionStatus.KNOWN
        assert alice_rec.person_id.startswith("ALICE")

        # Bob query
        bob_aligned = aligner.align(bob_images[0], detection)
        bob_emb = embedder.generate(bob_aligned)
        bob_rec = recognizer.recognize(bob_emb)
        assert bob_rec.status is RecognitionStatus.KNOWN
        assert bob_rec.person_id.startswith("BOB")

    def test_unknown_after_registration(
        self, service, recognizer,
    ):
        """
        After registration, unregistered faces should still be UNKNOWN.
        """
        # Register one person
        images = [_make_face_image(seed=42) for _ in range(5)]
        result = service.register("EMP-001", "Alice", images)
        assert result.status is RegistrationStatus.SUCCESS

        # Query with a completely different embedding
        unknown_emb = _make_embedding(seed=9999)
        rec_result = recognizer.recognize(unknown_emb)
        assert rec_result.status is RecognitionStatus.UNKNOWN

    def test_registration_lifecycle(
        self, service, store, recognizer,
    ):
        """
        Complete lifecycle: register → verify storage → recognize →
        verify result structure.
        """
        images = [_make_face_image(seed=i) for i in range(5)]
        result = service.register(
            person_id="LIFECYCLE-001",
            display_name="Test User",
            face_images=images,
            metadata={"department": "engineering"},
        )

        # Verify result structure
        assert isinstance(result, RegistrationResult)
        assert result.person_id == "LIFECYCLE-001"
        assert result.status is RegistrationStatus.SUCCESS
        assert result.accepted_count >= 3
        assert result.rejected_count >= 0
        assert result.accepted_count + result.rejected_count == len(images)
        assert len(result.sample_results) == len(images)
        assert result.timestamp is not None
        assert result.duplicate_person_id is None

        # Verify storage has metadata
        records = store.get_all()
        assert len(records) == result.accepted_count
        for record in records:
            assert "department" in record.metadata

    def test_duplicate_prevents_second_registration(
        self, store, aligner, recognizer, reg_config,
    ):
        """
        Registering a second person with the same face should be blocked.
        """
        # Both "people" will generate the same embedding because we use
        # the same face images
        same_images = [_make_face_image(seed=42) for _ in range(5)]

        svc1 = RegistrationService(
            detector=MockDetector(),
            aligner=aligner,
            embedder=MockEmbedder(),
            store=store,
            recognizer=recognizer,
            config=reg_config,
            embedding_dim=512,
        )

        # First registration — should succeed
        result1 = svc1.register("PERSON-A", "Alice", same_images)
        assert result1.status is RegistrationStatus.SUCCESS

        # Second registration with same face — should be DUPLICATE
        result2 = svc1.register("PERSON-B", "Not Alice", same_images)
        assert result2.status is RegistrationStatus.DUPLICATE
        assert result2.duplicate_person_id is not None

    def test_phase3_backward_compatibility(self, store, recognizer):
        """
        Embeddings stored directly (Phase 3 style) should still work
        alongside Phase 4 registered embeddings.
        """
        # Phase 3 style: store directly
        direct_emb = _make_embedding(seed=77)
        store.save(EmbeddingRecord(
            person_id="phase3_person",
            embedding=direct_emb,
        ))
        recognizer.refresh_gallery()

        # Recognize Phase 3 embedding
        result = recognizer.recognize(direct_emb)
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "phase3_person"
