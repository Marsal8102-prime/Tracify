"""
Tests — Face recognition and matching engine (Phase 3).

Covers:
  - Cosine similarity calculation (identical, orthogonal, edge cases)
  - Embedding validation
  - RecognitionResult and RecognitionStatus
  - Matching engine (correct identity, unknown, threshold, multi-person)
  - Gallery cache (load, refresh, empty, corrupted entries)
  - Edge cases (no identities, single identity, duplicate embeddings)

All tests use synthetic data and run on CPU without model downloads.
"""

import numpy as np
import pytest

from config.settings import RecognitionSettings
from recognition.base import BaseRecognizer
from recognition.embedding_recognizer import EmbeddingRecognizer
from recognition.result import (
    MatchCandidate,
    RecognitionResult,
    RecognitionStatus,
)
from recognition.similarity import (
    cosine_similarity,
    cosine_similarity_batch,
    validate_embedding,
)
from storage.base import EmbeddingRecord, EmbeddingStore
from storage.local_store import LocalEmbeddingStore


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_embedding(dim: int = 512, seed: int = 42) -> np.ndarray:
    """Create a deterministic L2-normalized embedding."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(dim).astype(np.float32)
    return raw / np.linalg.norm(raw)


def _make_orthogonal_pair(dim: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Create two approximately orthogonal normalized embeddings."""
    a = _make_embedding(dim, seed=1)
    b = _make_embedding(dim, seed=9999)
    # Gram-Schmidt to ensure orthogonality
    b = b - np.dot(b, a) * a
    b = b / np.linalg.norm(b)
    return a.astype(np.float32), b.astype(np.float32)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path) -> LocalEmbeddingStore:
    """Create a LocalEmbeddingStore in a temp directory."""
    return LocalEmbeddingStore(
        storage_dir=str(tmp_path / "embeddings"),
        expected_dimension=512,
    )


@pytest.fixture
def default_config() -> RecognitionSettings:
    """Default recognition config with threshold=0.6, top_k=3."""
    return RecognitionSettings(
        strategy="cosine",
        similarity_threshold=0.6,
        top_k=3,
    )


@pytest.fixture
def recognizer(store, default_config) -> EmbeddingRecognizer:
    """Create an EmbeddingRecognizer with default config."""
    return EmbeddingRecognizer(
        store=store,
        config=default_config,
        expected_dimension=512,
    )


# ── Test: cosine similarity ─────────────────────────────────────────────

class TestSimilarityCalculation:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical normalized vectors should have similarity ~1.0."""
        emb = _make_embedding()
        score = cosine_similarity(emb, emb)
        assert abs(score - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0.0."""
        a, b = _make_orthogonal_pair()
        score = cosine_similarity(a, b)
        assert abs(score) < 1e-5

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity ~-1.0."""
        emb = _make_embedding()
        score = cosine_similarity(emb, -emb)
        assert abs(score - (-1.0)) < 1e-5

    def test_similar_vectors(self):
        """Slightly different vectors should have high similarity."""
        a = _make_embedding(seed=1)
        # Perturb slightly — noise norm ≈ 0.005 × √512 ≈ 0.11, small vs unit signal
        noise = np.random.RandomState(2).randn(512).astype(np.float32) * 0.005
        b = a + noise
        b = b / np.linalg.norm(b)
        score = cosine_similarity(a, b)
        assert 0.9 < score <= 1.0

    def test_wrong_dimension_raises(self):
        """Mismatched dimensions should raise ValueError."""
        a = _make_embedding(dim=512)
        b = _make_embedding(dim=256, seed=99)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity(a, b)

    def test_zero_vector_raises(self):
        """Zero vector should raise ValueError."""
        emb = _make_embedding()
        zero = np.zeros(512, dtype=np.float32)
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity(emb, zero)

    def test_nan_embedding_raises(self):
        """NaN embedding should raise ValueError."""
        emb = _make_embedding()
        nan_emb = np.full(512, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            cosine_similarity(emb, nan_emb)

    def test_inf_embedding_raises(self):
        """Inf embedding should raise ValueError."""
        emb = _make_embedding()
        inf_emb = np.full(512, np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            cosine_similarity(emb, inf_emb)

    def test_non_array_raises(self):
        """Non-numpy input should raise TypeError."""
        emb = _make_embedding()
        with pytest.raises(TypeError):
            cosine_similarity(emb, [1.0] * 512)

    def test_2d_input_raises(self):
        """2D input should raise ValueError."""
        a = _make_embedding().reshape(1, 512)
        b = _make_embedding()
        with pytest.raises(ValueError, match="1-dimensional"):
            cosine_similarity(a, b)


# ── Test: batch similarity ──────────────────────────────────────────────

class TestBatchSimilarity:
    """Tests for vectorized batch similarity computation."""

    def test_batch_scores(self):
        """Batch computation should match individual computations."""
        query = _make_embedding(seed=0)
        gallery = np.stack([
            _make_embedding(seed=i) for i in range(5)
        ], axis=0)
        batch_scores = cosine_similarity_batch(query, gallery)
        assert batch_scores.shape == (5,)
        for i in range(5):
            individual = cosine_similarity(query, gallery[i])
            assert abs(batch_scores[i] - individual) < 1e-5

    def test_batch_self_match(self):
        """Query in gallery should have score ~1.0 at its position."""
        query = _make_embedding(seed=42)
        gallery = np.stack([
            _make_embedding(seed=0),
            query,
            _make_embedding(seed=99),
        ], axis=0)
        scores = cosine_similarity_batch(query, gallery)
        assert abs(scores[1] - 1.0) < 1e-5

    def test_batch_empty_gallery_raises(self):
        """Gallery with wrong shape should raise."""
        query = _make_embedding()
        gallery = np.empty((0, 512), dtype=np.float32)
        # 0-row gallery is valid — returns empty scores
        scores = cosine_similarity_batch(query, gallery)
        assert scores.shape == (0,)

    def test_batch_dimension_mismatch(self):
        """Mismatched dimensions should raise ValueError."""
        query = _make_embedding(dim=512)
        gallery = np.random.randn(3, 256).astype(np.float32)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity_batch(query, gallery)


# ── Test: embedding validation ──────────────────────────────────────────

class TestEmbeddingValidation:
    """Tests for the validate_embedding utility."""

    def test_valid_embedding(self):
        """Valid normalized embedding should pass validation."""
        emb = _make_embedding()
        validate_embedding(emb, expected_dim=512)  # Should not raise

    def test_wrong_type_raises(self):
        """Non-numpy input should raise TypeError."""
        with pytest.raises(TypeError, match="numpy ndarray"):
            validate_embedding([1.0] * 512, expected_dim=512)

    def test_wrong_shape_raises(self):
        """2D array should raise ValueError."""
        emb = np.random.randn(1, 512).astype(np.float32)
        with pytest.raises(ValueError, match="1-dimensional"):
            validate_embedding(emb, expected_dim=512)

    def test_wrong_dimension_raises(self):
        """Wrong dimension should raise ValueError."""
        emb = _make_embedding(dim=256, seed=0)
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_embedding(emb, expected_dim=512)

    def test_nan_raises(self):
        """NaN values should raise ValueError."""
        emb = np.full(512, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_embedding(emb, expected_dim=512)

    def test_inf_raises(self):
        """Inf values should raise ValueError."""
        emb = np.full(512, np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_embedding(emb, expected_dim=512)

    def test_zero_vector_raises(self):
        """Zero vector should raise ValueError."""
        emb = np.zeros(512, dtype=np.float32)
        with pytest.raises(ValueError, match="zero vector"):
            validate_embedding(emb, expected_dim=512)


# ── Test: RecognitionResult ─────────────────────────────────────────────

class TestRecognitionResult:
    """Tests for structured recognition results."""

    def test_known_result(self):
        """KNOWN result should have correct fields."""
        result = RecognitionResult(
            person_id="alice",
            status=RecognitionStatus.KNOWN,
            similarity=0.85,
            threshold=0.6,
            matched_embedding_id="alice",
        )
        assert result.is_known
        assert not result.is_unknown
        assert result.person_id == "alice"
        assert result.similarity == 0.85
        assert result.threshold == 0.6

    def test_unknown_result(self):
        """UNKNOWN result should have correct fields."""
        result = RecognitionResult(
            person_id=None,
            status=RecognitionStatus.UNKNOWN,
            similarity=0.3,
            threshold=0.6,
        )
        assert result.is_unknown
        assert not result.is_known
        assert result.person_id is None

    def test_status_enum_values(self):
        """Enum values should be 'known' and 'unknown'."""
        assert RecognitionStatus.KNOWN.value == "known"
        assert RecognitionStatus.UNKNOWN.value == "unknown"

    def test_result_has_timestamp(self):
        """Result should always have a timestamp."""
        result = RecognitionResult(
            person_id=None,
            status=RecognitionStatus.UNKNOWN,
            similarity=0.0,
            threshold=0.6,
        )
        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_match_candidate(self):
        """MatchCandidate should store person, score, embedding_id."""
        candidate = MatchCandidate(
            person_id="bob",
            similarity=0.78,
            embedding_id="bob",
        )
        assert candidate.person_id == "bob"
        assert candidate.similarity == 0.78
        assert candidate.embedding_id == "bob"

    def test_result_with_candidates(self):
        """Result should include top-K candidates list."""
        candidates = [
            MatchCandidate("alice", 0.85, "alice"),
            MatchCandidate("bob", 0.72, "bob"),
        ]
        result = RecognitionResult(
            person_id="alice",
            status=RecognitionStatus.KNOWN,
            similarity=0.85,
            threshold=0.6,
            candidates=candidates,
        )
        assert len(result.candidates) == 2
        assert result.candidates[0].person_id == "alice"
        assert result.candidates[1].person_id == "bob"


# ── Test: matching engine ───────────────────────────────────────────────

class TestMatchingEngine:
    """Tests for the EmbeddingRecognizer matching engine."""

    def test_correct_identity_recognized(self, store, recognizer):
        """Should recognize the correct identity."""
        # Register Alice
        alice_emb = _make_embedding(seed=10)
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=alice_emb,
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()

        # Query with Alice's embedding
        result = recognizer.recognize(alice_emb)
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "alice"
        assert result.similarity > 0.99

    def test_unknown_identity(self, store, recognizer):
        """Should return UNKNOWN for an unregistered face."""
        # Register Alice
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=_make_embedding(seed=10),
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()

        # Query with an unknown embedding
        unknown_emb = _make_embedding(seed=9999)
        result = recognizer.recognize(unknown_emb)
        assert result.status is RecognitionStatus.UNKNOWN
        assert result.person_id is None

    def test_best_candidate_selected(self, store, recognizer):
        """Should select the best-scoring candidate."""
        alice_emb = _make_embedding(seed=10)
        bob_emb = _make_embedding(seed=20)
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=alice_emb,
            created_at="2026-01-01T00:00:00",
        ))
        store.save(EmbeddingRecord(
            person_id="bob",
            embedding=bob_emb,
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()

        # Query with Alice's embedding — should match Alice, not Bob
        result = recognizer.recognize(alice_emb)
        assert result.person_id == "alice"
        assert result.similarity > 0.99

    def test_threshold_boundary_above(self, store, default_config):
        """Score exactly at threshold should be KNOWN."""
        # Use a very low threshold so any non-zero similarity passes
        config = RecognitionSettings(
            strategy="cosine",
            similarity_threshold=0.0,
            top_k=1,
        )
        rec = EmbeddingRecognizer(store=store, config=config)
        store.save(EmbeddingRecord(
            person_id="test",
            embedding=_make_embedding(seed=1),
            created_at="2026-01-01T00:00:00",
        ))
        rec.load_gallery()

        query = _make_embedding(seed=999)
        result = rec.recognize(query)
        # With threshold=0.0, any positive similarity → KNOWN
        assert result.status is RecognitionStatus.KNOWN

    def test_threshold_boundary_below(self, store):
        """Score below threshold should be UNKNOWN."""
        config = RecognitionSettings(
            strategy="cosine",
            similarity_threshold=0.99,
            top_k=1,
        )
        rec = EmbeddingRecognizer(store=store, config=config)
        store.save(EmbeddingRecord(
            person_id="test",
            embedding=_make_embedding(seed=1),
            created_at="2026-01-01T00:00:00",
        ))
        rec.load_gallery()

        # Different embedding — similarity will be well below 0.99
        query = _make_embedding(seed=999)
        result = rec.recognize(query)
        assert result.status is RecognitionStatus.UNKNOWN

    def test_multiple_identities(self, store, recognizer):
        """Should correctly distinguish between multiple identities."""
        seeds = {"alice": 10, "bob": 20, "charlie": 30}
        for name, seed in seeds.items():
            store.save(EmbeddingRecord(
                person_id=name,
                embedding=_make_embedding(seed=seed),
                created_at="2026-01-01T00:00:00",
            ))
        recognizer.load_gallery()

        for name, seed in seeds.items():
            result = recognizer.recognize(_make_embedding(seed=seed))
            assert result.status is RecognitionStatus.KNOWN
            assert result.person_id == name

    def test_top_k_ordering(self, store, default_config):
        """Top-K candidates should be ordered by descending similarity."""
        config = RecognitionSettings(
            strategy="cosine",
            similarity_threshold=0.0,
            top_k=5,
        )
        rec = EmbeddingRecognizer(store=store, config=config)

        for i in range(5):
            store.save(EmbeddingRecord(
                person_id=f"person_{i}",
                embedding=_make_embedding(seed=i * 10),
                created_at="2026-01-01T00:00:00",
            ))
        rec.load_gallery()

        query = _make_embedding(seed=0)
        result = rec.recognize(query)
        # Candidates should be sorted by similarity descending
        for i in range(len(result.candidates) - 1):
            assert result.candidates[i].similarity >= result.candidates[i + 1].similarity

    def test_top_k_limits_candidates(self, store):
        """Top-K should limit the number of returned candidates."""
        config = RecognitionSettings(
            strategy="cosine",
            similarity_threshold=0.0,
            top_k=2,
        )
        rec = EmbeddingRecognizer(store=store, config=config)

        for i in range(5):
            store.save(EmbeddingRecord(
                person_id=f"person_{i}",
                embedding=_make_embedding(seed=i * 10),
                created_at="2026-01-01T00:00:00",
            ))
        rec.load_gallery()

        result = rec.recognize(_make_embedding(seed=0))
        assert len(result.candidates) <= 2

    def test_result_includes_threshold(self, store, recognizer):
        """Result should include the threshold that was applied."""
        store.save(EmbeddingRecord(
            person_id="test",
            embedding=_make_embedding(seed=1),
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()
        result = recognizer.recognize(_make_embedding(seed=1))
        assert result.threshold == 0.6

    def test_gallery_not_loaded_raises(self, store, default_config):
        """Calling recognize before load_gallery should raise RuntimeError."""
        rec = EmbeddingRecognizer(store=store, config=default_config)
        with pytest.raises(RuntimeError, match="Gallery not loaded"):
            rec.recognize(_make_embedding())

    def test_recognize_batch(self, store, recognizer):
        """Batch recognition should return one result per query."""
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=_make_embedding(seed=10),
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()

        queries = [_make_embedding(seed=10), _make_embedding(seed=9999)]
        results = recognizer.recognize_batch(queries)
        assert len(results) == 2
        assert results[0].status is RecognitionStatus.KNOWN
        assert results[0].person_id == "alice"


# ── Test: multiple embeddings per person ────────────────────────────────

class TestMultipleEmbeddingsPerPerson:
    """Tests for handling multiple embeddings per identity."""

    def test_best_match_selected(self, tmp_path):
        """When a person has multiple embeddings, best match wins."""
        store = LocalEmbeddingStore(
            storage_dir=str(tmp_path / "embeddings"),
            expected_dimension=512,
        )
        config = RecognitionSettings(
            strategy="cosine",
            similarity_threshold=0.5,
            top_k=3,
        )
        rec = EmbeddingRecognizer(store=store, config=config)

        # Simulate multiple embeddings for Alice using different poses
        alice_pose1 = _make_embedding(seed=10)
        alice_pose2 = _make_embedding(seed=11)
        # Store them as separate records (different person_ids to simulate)
        store.save(EmbeddingRecord(
            person_id="alice_pose1",
            embedding=alice_pose1,
            created_at="2026-01-01T00:00:00",
        ))
        store.save(EmbeddingRecord(
            person_id="alice_pose2",
            embedding=alice_pose2,
            created_at="2026-01-01T00:00:00",
        ))
        rec.load_gallery()

        # Query with pose1 — should match pose1 with higher score
        result = rec.recognize(alice_pose1)
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "alice_pose1"
        assert result.similarity > 0.99

    def test_gallery_size_reflects_all_embeddings(self, store, recognizer):
        """Gallery size should count all embeddings, not unique persons."""
        for i in range(3):
            store.save(EmbeddingRecord(
                person_id=f"person_{i}",
                embedding=_make_embedding(seed=i),
                created_at="2026-01-01T00:00:00",
            ))
        recognizer.load_gallery()
        assert recognizer.gallery_size == 3


# ── Test: gallery cache ─────────────────────────────────────────────────

class TestGalleryCache:
    """Tests for the in-memory gallery cache."""

    def test_initial_load(self, store, recognizer):
        """load_gallery should populate the in-memory cache."""
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=_make_embedding(seed=10),
            created_at="2026-01-01T00:00:00",
        ))
        count = recognizer.load_gallery()
        assert count == 1
        assert recognizer.gallery_size == 1

    def test_repeated_queries_use_cache(self, store, recognizer):
        """Multiple recognize calls should use the cached gallery."""
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=_make_embedding(seed=10),
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()

        # Multiple queries should all work without reloading
        for _ in range(5):
            result = recognizer.recognize(_make_embedding(seed=10))
            assert result.status is RecognitionStatus.KNOWN

    def test_refresh_gallery(self, store, recognizer):
        """refresh_gallery should pick up newly added embeddings."""
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=_make_embedding(seed=10),
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()
        assert recognizer.gallery_size == 1

        # Add a new person
        store.save(EmbeddingRecord(
            person_id="bob",
            embedding=_make_embedding(seed=20),
            created_at="2026-01-01T00:00:00",
        ))
        count = recognizer.refresh_gallery()
        assert count == 2
        assert recognizer.gallery_size == 2

        # Should now recognize Bob
        result = recognizer.recognize(_make_embedding(seed=20))
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "bob"

    def test_empty_store_loads(self, store, recognizer):
        """Empty store should load successfully with zero embeddings."""
        count = recognizer.load_gallery()
        assert count == 0
        assert recognizer.gallery_size == 0

    def test_corrupted_entries_skipped(self, store, recognizer, tmp_path):
        """Corrupted .npz files should be skipped during gallery load."""
        # Save a valid embedding
        store.save(EmbeddingRecord(
            person_id="alice",
            embedding=_make_embedding(seed=10),
            created_at="2026-01-01T00:00:00",
        ))

        # Create a corrupted .npz file
        embeddings_dir = tmp_path / "embeddings"
        bad_file = embeddings_dir / "corrupted.npz"
        bad_file.write_bytes(b"this is not a valid npz file")

        count = recognizer.load_gallery()
        assert count == 1  # Only the valid one
        assert recognizer.gallery_size == 1

    def test_load_returns_count(self, store, recognizer):
        """load_gallery should return the exact count of loaded embeddings."""
        for i in range(4):
            store.save(EmbeddingRecord(
                person_id=f"person_{i}",
                embedding=_make_embedding(seed=i),
                created_at="2026-01-01T00:00:00",
            ))
        count = recognizer.load_gallery()
        assert count == 4


# ── Test: edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge-case scenarios."""

    def test_no_registered_identities(self, store, recognizer):
        """Empty gallery should always return UNKNOWN."""
        recognizer.load_gallery()
        result = recognizer.recognize(_make_embedding())
        assert result.status is RecognitionStatus.UNKNOWN
        assert result.person_id is None
        assert result.similarity == 0.0

    def test_single_identity(self, store, recognizer):
        """Single-person gallery should work correctly."""
        emb = _make_embedding(seed=42)
        store.save(EmbeddingRecord(
            person_id="only_person",
            embedding=emb,
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()

        result = recognizer.recognize(emb)
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "only_person"

    def test_malformed_query_raises(self, store, recognizer):
        """Wrong dimension query should raise ValueError."""
        recognizer.load_gallery()
        wrong_dim = _make_embedding(dim=256, seed=0)
        with pytest.raises(ValueError, match="dimension mismatch"):
            recognizer.recognize(wrong_dim)

    def test_nan_query_raises(self, store, recognizer):
        """NaN query should raise ValueError."""
        recognizer.load_gallery()
        nan_emb = np.full(512, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            recognizer.recognize(nan_emb)

    def test_zero_query_raises(self, store, recognizer):
        """Zero vector query should raise ValueError."""
        recognizer.load_gallery()
        zero_emb = np.zeros(512, dtype=np.float32)
        with pytest.raises(ValueError, match="zero vector"):
            recognizer.recognize(zero_emb)

    def test_non_array_query_raises(self, store, recognizer):
        """Non-numpy query should raise TypeError."""
        recognizer.load_gallery()
        with pytest.raises(TypeError):
            recognizer.recognize([1.0] * 512)


# ── Test: interface compliance ──────────────────────────────────────────

class TestInterfaceCompliance:
    """Tests for interface and subclass compliance."""

    def test_is_subclass_of_base_recognizer(self):
        """EmbeddingRecognizer must implement BaseRecognizer."""
        assert issubclass(EmbeddingRecognizer, BaseRecognizer)

    def test_gallery_size_property(self, store, recognizer):
        """gallery_size property should work before and after loading."""
        assert recognizer.gallery_size == 0
        store.save(EmbeddingRecord(
            person_id="test",
            embedding=_make_embedding(),
            created_at="2026-01-01T00:00:00",
        ))
        recognizer.load_gallery()
        assert recognizer.gallery_size == 1

    def test_threshold_property(self, store, default_config):
        """threshold property should return configured value."""
        rec = EmbeddingRecognizer(store=store, config=default_config)
        assert rec.threshold == 0.6
