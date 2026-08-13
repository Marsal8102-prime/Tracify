"""
Tests — End-to-end Phase 3 recognition integration test.

Verifies the complete chain:
    Frame → Alignment → Mock Embedding → Storage → Recognition → Result

Reuses Phase 1 and Phase 2 components (FaceAligner, LocalEmbeddingStore,
EmbeddingRecord) and adds the Phase 3 recognition layer.

All tests use synthetic data and mock embeddings to avoid requiring
a GPU or model download.
"""

import numpy as np
import pytest

from alignment import FaceAligner
from config.settings import AlignmentSettings, RecognitionSettings
from detection.base import DetectionResult
from recognition import (
    EmbeddingRecognizer,
    RecognitionResult,
    RecognitionStatus,
)
from storage import EmbeddingRecord, LocalEmbeddingStore


# ── Helpers ─────────────────────────────────────────────────────────────

def _mock_generate_embedding(
    aligned_face: np.ndarray,
    dimension: int = 512,
) -> np.ndarray:
    """
    Deterministic mock embedding generator.
    Produces a normalized 512-d vector from image pixel statistics.
    """
    rng = np.random.RandomState(int(aligned_face.mean() * 1000) % (2**31))
    raw = rng.randn(dimension).astype(np.float32)
    norm = np.linalg.norm(raw)
    if norm < 1e-10:
        return np.zeros(dimension, dtype=np.float32)
    return raw / norm


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def aligner() -> FaceAligner:
    config = AlignmentSettings(output_size=[112, 112])
    return FaceAligner(config)


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
    return EmbeddingRecognizer(
        store=store,
        config=config,
        expected_dimension=512,
    )


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """A 720p synthetic frame with a bright face-like region."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[200:400, 500:700, :] = 180
    return frame


@pytest.fixture
def simulated_detection() -> DetectionResult:
    """Simulated detection result with landmarks in the bright region."""
    return DetectionResult(
        bbox=np.array([500.0, 200.0, 700.0, 400.0], dtype=np.float32),
        confidence=0.92,
        landmarks=np.array(
            [
                [550.0, 270.0],
                [650.0, 270.0],
                [600.0, 320.0],
                [560.0, 370.0],
                [640.0, 370.0],
            ],
            dtype=np.float32,
        ),
    )


# ── End-to-end tests ────────────────────────────────────────────────────

class TestPhase3EndToEnd:
    """
    End-to-end test of the full pipeline through recognition:
    Frame → Align → Embed (mock) → Store → Recognize → Result
    """

    def test_full_pipeline_known(
        self,
        aligner,
        store,
        recognizer,
        synthetic_frame,
        simulated_detection,
    ):
        """
        Complete pipeline: register a known face, then query it.
        Should return KNOWN status.
        """
        # Step 1: Align the face
        aligned = aligner.align(synthetic_frame, simulated_detection)
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)

        # Step 2: Generate mock embedding
        embedding = _mock_generate_embedding(aligned)
        assert embedding.shape == (512,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5

        # Step 3: Register as known person
        store.save(EmbeddingRecord(
            person_id="known_employee",
            embedding=embedding,
            metadata={"model": "arcface", "source": "e2e_test"},
            created_at="2026-08-09T12:00:00",
        ))

        # Step 4: Load gallery and recognize
        recognizer.load_gallery()
        assert recognizer.gallery_size == 1

        # Step 5: Query with the same face (same frame → same embedding)
        query = _mock_generate_embedding(aligned)
        result = recognizer.recognize(query)

        # Step 6: Verify result
        assert isinstance(result, RecognitionResult)
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "known_employee"
        assert result.similarity > 0.99
        assert result.threshold == 0.6
        assert result.is_known
        assert not result.is_unknown

    def test_full_pipeline_unknown(
        self,
        aligner,
        store,
        recognizer,
        synthetic_frame,
        simulated_detection,
    ):
        """
        Register a known face, then query with a different face.
        Should return UNKNOWN status.
        """
        # Register known person with a specific embedding
        known_emb = np.random.RandomState(42).randn(512).astype(np.float32)
        known_emb = known_emb / np.linalg.norm(known_emb)
        store.save(EmbeddingRecord(
            person_id="registered_person",
            embedding=known_emb,
            created_at="2026-08-09T12:00:00",
        ))

        # Align a face (produces a different embedding than the registered one)
        aligned = aligner.align(synthetic_frame, simulated_detection)
        query_emb = _mock_generate_embedding(aligned)

        # Recognize
        recognizer.load_gallery()
        result = recognizer.recognize(query_emb)

        # Should be unknown (different embedding than registered)
        assert result.status is RecognitionStatus.UNKNOWN
        assert result.person_id is None
        assert result.is_unknown

    def test_multiple_registered_faces(
        self,
        store,
        recognizer,
    ):
        """
        Register multiple known faces, query each one.
        Each should be correctly identified.
        """
        persons = {
            "alice": np.random.RandomState(100),
            "bob": np.random.RandomState(200),
            "charlie": np.random.RandomState(300),
        }

        embeddings = {}
        for name, rng in persons.items():
            emb = rng.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings[name] = emb
            store.save(EmbeddingRecord(
                person_id=name,
                embedding=emb,
                created_at="2026-08-09T12:00:00",
            ))

        recognizer.load_gallery()
        assert recognizer.gallery_size == 3

        # Each query should match the correct person
        for name, emb in embeddings.items():
            result = recognizer.recognize(emb)
            assert result.status is RecognitionStatus.KNOWN
            assert result.person_id == name
            assert result.similarity > 0.99

    def test_pipeline_with_store_verification(
        self,
        aligner,
        store,
        recognizer,
        synthetic_frame,
        simulated_detection,
    ):
        """
        Verify that recognition integrates properly with storage:
        save → load gallery → recognize → verify store state.
        """
        aligned = aligner.align(synthetic_frame, simulated_detection)
        embedding = _mock_generate_embedding(aligned)

        # Save and verify storage
        store.save(EmbeddingRecord(
            person_id="employee_001",
            embedding=embedding,
            metadata={"department": "engineering"},
            created_at="2026-08-09T12:00:00",
        ))
        assert store.count() == 1
        assert "employee_001" in store.list_ids()

        # Load gallery and verify
        count = recognizer.load_gallery()
        assert count == 1

        # Recognize
        result = recognizer.recognize(embedding)
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "employee_001"

        # Verify stored record is intact
        loaded = store.get("employee_001")
        assert loaded is not None
        assert loaded.metadata["department"] == "engineering"

    def test_empty_gallery_returns_unknown(self, recognizer):
        """Empty gallery should always return UNKNOWN."""
        recognizer.load_gallery()
        assert recognizer.gallery_size == 0

        emb = np.random.RandomState(42).randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        result = recognizer.recognize(emb)
        assert result.status is RecognitionStatus.UNKNOWN
        assert result.similarity == 0.0
