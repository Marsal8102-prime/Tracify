"""
Tests — Phase 3F identity fix.

Verifies the invariant that EmbeddingRecord.person_id always carries
the canonical business person_id, while storage filenames use the
composite storage_key.

Covers:
  a. canonical person_id is stored in EmbeddingRecord
  b. multiple embeddings have unique storage keys/files
  c. gallery grouping treats all embeddings as the same person
  d. recognition returns canonical person_id
  e. matched_embedding_id remains the individual embedding identifier
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from alignment.base import BaseAligner
from config.settings import RecognitionSettings, RegistrationSettings
from detection.base import BaseDetector, DetectionResult
from embedding.base import BaseEmbedder
from recognition import EmbeddingRecognizer
from recognition.result import RecognitionStatus
from registration.result import RegistrationStatus
from registration.service import RegistrationService, _generate_embedding_key
from storage import EmbeddingRecord, LocalEmbeddingStore


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_embedding(dim: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    raw = rng.randn(dim).astype(np.float32)
    return raw / np.linalg.norm(raw)


def _make_face_image(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_detection(confidence: float = 0.95, face_size: float = 150.0) -> DetectionResult:
    return DetectionResult(
        bbox=np.array([100, 80, 100 + face_size, 80 + face_size], dtype=np.float32),
        confidence=confidence,
        landmarks=np.array(
            [[140.0, 140.0], [210.0, 140.0], [175.0, 180.0],
             [145.0, 220.0], [205.0, 220.0]], dtype=np.float32,
        ),
    )


# ── Mock components ────────────────────────────────────────────────────

class _MockDetector(BaseDetector):
    def __init__(self):
        self._loaded = True

    def load_model(self): pass

    def detect(self, frame):
        return [_make_detection()]

    @property
    def is_loaded(self):
        return self._loaded


class _MockAligner(BaseAligner):
    def align(self, frame, detection):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    @property
    def output_size(self):
        return (112, 112)


class _MockEmbedder(BaseEmbedder):
    """Returns distinct but deterministic embeddings per call."""

    def __init__(self):
        self._loaded = True
        self._call_count = 0

    def load_model(self): pass

    def generate(self, aligned_face):
        self._call_count += 1
        return _make_embedding(seed=self._call_count * 7)

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def dimension(self):
        return 512


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
        top_k=5,
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
        minimum_samples=1,
        maximum_samples=5,
        minimum_face_size=80,
        duplicate_threshold=0.99,  # high threshold so dup check doesn't interfere
        quality_checks_enabled=True,
    )


@pytest.fixture
def service(store, recognizer, reg_config) -> RegistrationService:
    return RegistrationService(
        detector=_MockDetector(),
        aligner=_MockAligner(),
        embedder=_MockEmbedder(),
        store=store,
        recognizer=recognizer,
        config=reg_config,
        embedding_dim=512,
    )


# ── (a) Canonical person_id stored in EmbeddingRecord ───────────────────

class TestCanonicalPersonIdStored:
    """Verify that every persisted EmbeddingRecord carries the canonical
    business person_id, not a composite storage key."""

    def test_record_person_id_is_canonical(self, service, store):
        """After registration, get_all() records should have the
        canonical person_id, not a composite key."""
        images = [_make_face_image(seed=i) for i in range(3)]
        result = service.register("PERSON-A", "Person A", images)
        assert result.status is RegistrationStatus.SUCCESS

        records = store.get_all()
        assert len(records) == result.accepted_count
        for record in records:
            assert record.person_id == "PERSON-A"
            assert "__emb_" not in record.person_id

    def test_storage_key_field_on_creation(self):
        """EmbeddingRecord with storage_key should carry both fields
        independently."""
        emb = _make_embedding(seed=1)
        record = EmbeddingRecord(
            person_id="canonical_id",
            embedding=emb,
            storage_key="canonical_id__emb_0",
        )
        assert record.person_id == "canonical_id"
        assert record.storage_key == "canonical_id__emb_0"

    def test_storage_key_defaults_to_none(self):
        """Without explicit storage_key, field should default to None."""
        record = EmbeddingRecord(
            person_id="someone",
            embedding=_make_embedding(),
        )
        assert record.storage_key is None


# ── (b) Unique storage keys/files for multi-embedding ────────────────────

class TestUniqueStorageFiles:
    """Verify that multi-embedding registration produces unique files
    on disk, each named with the composite storage key."""

    def test_unique_files_per_embedding(self, service, store, tmp_path):
        """Each accepted embedding should produce a unique .npz file."""
        images = [_make_face_image(seed=i) for i in range(3)]
        result = service.register("MULTI-EMB", "Multi Emb", images)
        assert result.status is RegistrationStatus.SUCCESS

        embeddings_dir = tmp_path / "embeddings"
        npz_files = sorted(embeddings_dir.glob("*.npz"))
        assert len(npz_files) == result.accepted_count

        # Each file should have a distinct name using the composite key
        names = [f.stem for f in npz_files]
        assert len(set(names)) == result.accepted_count
        for name in names:
            assert name.startswith("MULTI-EMB__emb_")

    def test_storage_key_only_affects_filename(self, store, tmp_path):
        """save() with storage_key should use it for the filename but
        persist person_id inside the .npz."""
        emb = _make_embedding(seed=10)
        store.save(EmbeddingRecord(
            person_id="real_id",
            embedding=emb,
            storage_key="real_id__emb_0",
        ))

        # File on disk should use storage_key
        embeddings_dir = tmp_path / "embeddings"
        npz_files = list(embeddings_dir.glob("*.npz"))
        assert len(npz_files) == 1
        assert npz_files[0].stem == "real_id__emb_0"

        # Loaded record should have canonical person_id
        records = store.get_all()
        assert len(records) == 1
        assert records[0].person_id == "real_id"

    def test_rollback_deletes_by_storage_key(self, store, tmp_path):
        """delete() with composite key should remove the correct file."""
        emb = _make_embedding(seed=10)
        store.save(EmbeddingRecord(
            person_id="person_x",
            embedding=emb,
            storage_key="person_x__emb_0",
        ))
        assert store.count() == 1

        # Delete by storage key (as _rollback() does)
        deleted = store.delete("person_x__emb_0")
        assert deleted is True
        assert store.count() == 0


# ── (c) Gallery grouping treats all embeddings as one person ─────────────

class TestGalleryGrouping:
    """Verify that the recognizer groups all embeddings of the same
    canonical person_id into a single candidate."""

    def test_multi_embedding_grouped_as_one_person(self, store, recognizer):
        """3 embeddings with the same person_id should produce a single
        person in the gallery's candidate list."""
        for i in range(3):
            store.save(EmbeddingRecord(
                person_id="GROUPED-PERSON",
                embedding=_make_embedding(seed=100 + i),
                storage_key=f"GROUPED-PERSON__emb_{i}",
            ))

        count = recognizer.load_gallery()
        assert count == 3

        # Query with one of the registered embeddings
        query = _make_embedding(seed=100)
        result = recognizer.recognize(query)

        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "GROUPED-PERSON"

        # Only ONE candidate for this person (grouping worked)
        person_ids_in_candidates = [c.person_id for c in result.candidates]
        assert person_ids_in_candidates.count("GROUPED-PERSON") == 1

    def test_two_persons_remain_separate(self, store, recognizer):
        """Embeddings for different persons should remain separate
        candidates, not merged."""
        for i in range(2):
            store.save(EmbeddingRecord(
                person_id="ALICE",
                embedding=_make_embedding(seed=10 + i),
                storage_key=f"ALICE__emb_{i}",
            ))
        for i in range(2):
            store.save(EmbeddingRecord(
                person_id="BOB",
                embedding=_make_embedding(seed=500 + i),
                storage_key=f"BOB__emb_{i}",
            ))

        count = recognizer.load_gallery()
        assert count == 4

        query = _make_embedding(seed=10)
        result = recognizer.recognize(query)

        # Should find ALICE as top match
        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "ALICE"

        # Candidates should have both ALICE and BOB — as 2 separate entries
        candidate_ids = {c.person_id for c in result.candidates}
        assert "ALICE" in candidate_ids
        assert "BOB" in candidate_ids


# ── (d) Recognition returns canonical person_id ──────────────────────────

class TestRecognitionReturnsCanonicalId:
    """Verify that the recognize() result carries the canonical
    person_id, not a composite storage key."""

    def test_recognized_person_id_is_canonical(self, service, store, recognizer):
        """After registration + gallery refresh, recognition should
        return the canonical person_id."""
        images = [_make_face_image(seed=i) for i in range(3)]
        result = service.register("CANONICAL-001", "Canonical", images)
        assert result.status is RegistrationStatus.SUCCESS

        # The registration service already refreshes the gallery.
        # Query with the first stored embedding.
        records = store.get_all()
        query = records[0].embedding
        rec_result = recognizer.recognize(query)

        assert rec_result.status is RecognitionStatus.KNOWN
        assert rec_result.person_id == "CANONICAL-001"
        assert "__emb_" not in rec_result.person_id


# ── (e) matched_embedding_id is the per-embedding identifier ─────────────

class TestMatchedEmbeddingId:
    """Verify that matched_embedding_id is a per-embedding identifier
    distinct from the canonical person_id."""

    def test_matched_embedding_id_differs_from_person_id(self, store, recognizer):
        """When multiple embeddings exist for a person, the
        matched_embedding_id should distinguish which one matched."""
        for i in range(3):
            store.save(EmbeddingRecord(
                person_id="PERSON-X",
                embedding=_make_embedding(seed=200 + i),
                storage_key=f"PERSON-X__emb_{i}",
            ))

        recognizer.load_gallery()
        query = _make_embedding(seed=200)  # matches first embedding exactly
        result = recognizer.recognize(query)

        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "PERSON-X"
        assert result.matched_embedding_id is not None
        # The embedding_id should be the gallery's internal identifier,
        # NOT the composite storage key (gallery builds its own IDs)
        # But it must start with the canonical person_id
        assert result.matched_embedding_id.startswith("PERSON-X")

    def test_single_embedding_matched_id_equals_person_id(self, store, recognizer):
        """With only one embedding, matched_embedding_id should equal
        person_id (no disambiguation suffix needed)."""
        store.save(EmbeddingRecord(
            person_id="SOLO",
            embedding=_make_embedding(seed=999),
            storage_key="SOLO__emb_0",
        ))
        recognizer.load_gallery()
        result = recognizer.recognize(_make_embedding(seed=999))

        assert result.status is RecognitionStatus.KNOWN
        assert result.person_id == "SOLO"
        assert result.matched_embedding_id == "SOLO"
