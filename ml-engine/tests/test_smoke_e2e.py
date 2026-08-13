"""
Tests — End-to-end smoke test for Phase 1 + Phase 2 pipeline.

Verifies the complete chain:
    Frame → Preprocessing → Detection → Alignment → Embedding → Storage

Uses synthetic data for the non-model-dependent parts and a mock
embedder to avoid requiring a GPU or large model download.

For full integration testing with a real InsightFace model, run
with: pytest -m integration
"""

import numpy as np
import pytest

from config.settings import (
    AlignmentSettings,
    DetectionSettings,
    EmbeddingSettings,
    PreprocessingSettings,
)
from detection.base import DetectionResult
from preprocessing import FramePreprocessor
from alignment import FaceAligner
from storage import LocalEmbeddingStore, EmbeddingRecord


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def preprocessor() -> FramePreprocessor:
    config = PreprocessingSettings(max_dimension=640, equalize_histogram=False)
    return FramePreprocessor(config)


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
def synthetic_frame() -> np.ndarray:
    """A 720p synthetic frame with some non-trivial content."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw a bright region where a "face" would be
    frame[200:400, 500:700, :] = 180
    return frame


@pytest.fixture
def simulated_detection() -> DetectionResult:
    """
    Simulated detection result as if SCRFD found a face.
    Landmarks are placed within the bright region of synthetic_frame.
    """
    return DetectionResult(
        bbox=np.array([500.0, 200.0, 700.0, 400.0], dtype=np.float32),
        confidence=0.92,
        landmarks=np.array(
            [
                [550.0, 270.0],  # left eye
                [650.0, 270.0],  # right eye
                [600.0, 320.0],  # nose
                [560.0, 370.0],  # left mouth
                [640.0, 370.0],  # right mouth
            ],
            dtype=np.float32,
        ),
    )


def _mock_generate_embedding(aligned_face: np.ndarray, dimension: int = 512) -> np.ndarray:
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


# ── Smoke test: complete pipeline ────────────────────────────────────────

class TestPhase2SmokeTest:
    """
    End-to-end test of the Phase 1 → Phase 2 pipeline using synthetic
    data and a mock embedder.
    """

    def test_full_pipeline(
        self,
        preprocessor,
        aligner,
        store,
        synthetic_frame,
        simulated_detection,
    ):
        """
        Complete pipeline: frame → preprocess → (simulated) detect →
        align → embed → store → retrieve.
        """
        # Step 1: Preprocessing
        processed = preprocessor.process(synthetic_frame)
        assert processed.frame is not None
        assert processed.scale_factor > 0

        # Step 2: Detection (simulated — using our fixture)
        # In real usage, detection would run on processed.frame.
        # Scale detection back to original coordinates.
        detection = simulated_detection  # already in original coords

        # Step 3: Alignment
        aligned = aligner.align(synthetic_frame, detection)
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)
        assert aligned.dtype == np.uint8

        # Step 4: Embedding (mock — avoids model dependency)
        embedding = _mock_generate_embedding(aligned)
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5

        # Step 5: Storage
        record = EmbeddingRecord(
            person_id="smoke_test_person",
            embedding=embedding,
            metadata={
                "model": "arcface",
                "version": "buffalo_l",
                "source": "smoke_test",
            },
        )
        store.save(record)

        # Step 6: Retrieval and verification
        loaded = store.get("smoke_test_person")
        assert loaded is not None
        assert loaded.person_id == "smoke_test_person"
        np.testing.assert_array_almost_equal(loaded.embedding, embedding)
        assert loaded.metadata["model"] == "arcface"

        # Verify store state
        assert store.count() == 1
        assert "smoke_test_person" in store.list_ids()

    def test_pipeline_with_scale_factor(
        self,
        preprocessor,
        aligner,
        store,
    ):
        """
        Test that scale_factor from preprocessing correctly maps
        detection coordinates back to original frame space.
        """
        # Large frame that will be resized
        large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        processed = preprocessor.process(large_frame)

        # Scale factor should be < 1.0 for frames larger than max_dimension
        assert processed.scale_factor < 1.0

        # Simulate a detection in the preprocessed frame space
        det_in_processed = DetectionResult(
            bbox=np.array([100, 80, 200, 180], dtype=np.float32),
            confidence=0.9,
            landmarks=np.array(
                [
                    [120.0, 110.0],
                    [180.0, 110.0],
                    [150.0, 140.0],
                    [125.0, 165.0],
                    [175.0, 165.0],
                ],
                dtype=np.float32,
            ),
        )

        # Scale back to original
        det_original = det_in_processed.scale_to_original(processed.scale_factor)

        # All coordinates should be larger in the original frame
        assert det_original.bbox[0] > det_in_processed.bbox[0]

        # Align using original frame and scaled-back coordinates
        aligned = aligner.align(large_frame, det_original)
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)

    def test_multiple_faces_pipeline(self, aligner, store):
        """Test pipeline handling multiple detected faces in one frame."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        detections = [
            DetectionResult(
                bbox=np.array([100, 100, 250, 300], dtype=np.float32),
                confidence=0.95,
                landmarks=np.array(
                    [[130, 160], [220, 160], [175, 200], [140, 250], [210, 250]],
                    dtype=np.float32,
                ),
            ),
            DetectionResult(
                bbox=np.array([500, 100, 650, 300], dtype=np.float32),
                confidence=0.88,
                landmarks=np.array(
                    [[530, 160], [620, 160], [575, 200], [540, 250], [610, 250]],
                    dtype=np.float32,
                ),
            ),
        ]

        for i, det in enumerate(detections):
            aligned = aligner.align(frame, det)
            assert aligned is not None

            embedding = _mock_generate_embedding(aligned)
            record = EmbeddingRecord(
                person_id=f"person_{i}",
                embedding=embedding,
                metadata={"model": "arcface"},
            )
            store.save(record)

        assert store.count() == 2
        assert len(store.get_all()) == 2
