"""
Tests — Embedding storage module.

Tests save/load, deletion, listing, dimension validation, corrupted
file handling, and edge cases. Uses a temporary directory so tests
don't affect real data.
"""

import numpy as np
import pytest

from storage.base import EmbeddingRecord, EmbeddingStore
from storage.local_store import LocalEmbeddingStore, _sanitize_filename


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path) -> LocalEmbeddingStore:
    """Create a LocalEmbeddingStore in a temp directory."""
    return LocalEmbeddingStore(
        storage_dir=str(tmp_path / "embeddings"),
        expected_dimension=512,
    )


@pytest.fixture
def sample_record() -> EmbeddingRecord:
    """Create a valid EmbeddingRecord with a normalized embedding."""
    raw = np.random.randn(512).astype(np.float32)
    embedding = raw / np.linalg.norm(raw)
    return EmbeddingRecord(
        person_id="john_doe_001",
        embedding=embedding,
        metadata={"model": "arcface", "version": "buffalo_l"},
        created_at="2026-08-08T14:00:00",
    )


# ── Test: save and load ─────────────────────────────────────────────────

class TestStorageSaveLoad:
    """Tests for saving and loading embeddings."""

    def test_save_and_get(self, store, sample_record):
        """Saved record should be retrievable."""
        store.save(sample_record)
        loaded = store.get(sample_record.person_id)
        assert loaded is not None
        assert loaded.person_id == sample_record.person_id
        np.testing.assert_array_almost_equal(loaded.embedding, sample_record.embedding)

    def test_metadata_preserved(self, store, sample_record):
        """Metadata should survive save/load cycle."""
        store.save(sample_record)
        loaded = store.get(sample_record.person_id)
        assert loaded is not None
        assert loaded.metadata == sample_record.metadata

    def test_created_at_preserved(self, store, sample_record):
        """created_at timestamp should survive save/load cycle."""
        store.save(sample_record)
        loaded = store.get(sample_record.person_id)
        assert loaded is not None
        assert loaded.created_at == sample_record.created_at

    def test_overwrite_existing(self, store, sample_record):
        """Saving with the same person_id should overwrite."""
        store.save(sample_record)
        new_embedding = np.random.randn(512).astype(np.float32)
        new_embedding = new_embedding / np.linalg.norm(new_embedding)
        updated = EmbeddingRecord(
            person_id=sample_record.person_id,
            embedding=new_embedding,
        )
        store.save(updated)
        loaded = store.get(sample_record.person_id)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded.embedding, new_embedding)

    def test_get_nonexistent(self, store):
        """Getting a non-existent ID should return None."""
        result = store.get("nonexistent_person")
        assert result is None

    def test_embedding_dtype_float32(self, store, sample_record):
        """Loaded embedding should be float32."""
        store.save(sample_record)
        loaded = store.get(sample_record.person_id)
        assert loaded is not None
        assert loaded.embedding.dtype == np.float32


# ── Test: delete ─────────────────────────────────────────────────────────

class TestStorageDelete:
    """Tests for embedding deletion."""

    def test_delete_existing(self, store, sample_record):
        """Deleting existing record should return True."""
        store.save(sample_record)
        result = store.delete(sample_record.person_id)
        assert result is True

    def test_delete_removes_record(self, store, sample_record):
        """Deleted record should no longer be retrievable."""
        store.save(sample_record)
        store.delete(sample_record.person_id)
        assert store.get(sample_record.person_id) is None

    def test_delete_nonexistent(self, store):
        """Deleting non-existent record should return False."""
        result = store.delete("nonexistent")
        assert result is False


# ── Test: list and count ─────────────────────────────────────────────────

class TestStorageListCount:
    """Tests for listing and counting embeddings."""

    def test_count_empty(self, store):
        """Empty store should have count 0."""
        assert store.count() == 0

    def test_count_after_saves(self, store):
        """Count should reflect number of saved embeddings."""
        for i in range(3):
            raw = np.random.randn(512).astype(np.float32)
            record = EmbeddingRecord(
                person_id=f"person_{i}",
                embedding=raw / np.linalg.norm(raw),
            )
            store.save(record)
        assert store.count() == 3

    def test_list_ids(self, store):
        """list_ids should return all stored person IDs."""
        ids = ["alice", "bob", "charlie"]
        for pid in ids:
            raw = np.random.randn(512).astype(np.float32)
            store.save(EmbeddingRecord(person_id=pid, embedding=raw / np.linalg.norm(raw)))
        stored_ids = store.list_ids()
        assert sorted(stored_ids) == sorted(ids)

    def test_get_all(self, store):
        """get_all should return all records."""
        for i in range(3):
            raw = np.random.randn(512).astype(np.float32)
            store.save(EmbeddingRecord(
                person_id=f"person_{i}",
                embedding=raw / np.linalg.norm(raw),
            ))
        records = store.get_all()
        assert len(records) == 3


# ── Test: validation ─────────────────────────────────────────────────────

class TestStorageValidation:
    """Tests for input validation."""

    def test_reject_wrong_dimension(self, store):
        """Should reject embedding with wrong dimension."""
        wrong_dim = np.random.randn(256).astype(np.float32)
        record = EmbeddingRecord(person_id="test", embedding=wrong_dim)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.save(record)

    def test_reject_none_embedding(self, store):
        """Should reject None embedding."""
        record = EmbeddingRecord(person_id="test", embedding=None)
        with pytest.raises(ValueError, match="not be None"):
            store.save(record)

    def test_reject_empty_person_id(self, store):
        """Should reject empty person_id."""
        raw = np.random.randn(512).astype(np.float32)
        record = EmbeddingRecord(person_id="", embedding=raw / np.linalg.norm(raw))
        with pytest.raises(ValueError, match="non-empty"):
            store.save(record)

    def test_reject_nan_embedding(self, store):
        """Should reject embedding containing NaN."""
        nan_emb = np.full(512, np.nan, dtype=np.float32)
        record = EmbeddingRecord(person_id="test", embedding=nan_emb)
        with pytest.raises(ValueError, match="NaN"):
            store.save(record)

    def test_reject_2d_embedding(self, store):
        """Should reject 2D embedding array."""
        emb_2d = np.random.randn(1, 512).astype(np.float32)
        record = EmbeddingRecord(person_id="test", embedding=emb_2d)
        with pytest.raises(ValueError, match="1-dimensional"):
            store.save(record)


# ── Test: corrupted file handling ────────────────────────────────────────

class TestStorageCorruptedFiles:
    """Tests for graceful handling of corrupted storage files."""

    def test_corrupted_file_returns_none(self, store, tmp_path):
        """Corrupted .npz file should return None on get, not crash."""
        # Write garbage to a .npz file
        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        bad_file = embeddings_dir / "bad_person.npz"
        bad_file.write_bytes(b"this is not a valid npz file")

        result = store.get("bad_person")
        assert result is None

    def test_corrupted_file_skipped_in_list(self, store, sample_record, tmp_path):
        """Corrupted files should be skipped in list_ids and get_all."""
        store.save(sample_record)

        # Add a corrupted file
        embeddings_dir = tmp_path / "embeddings"
        bad_file = embeddings_dir / "corrupted.npz"
        bad_file.write_bytes(b"garbage")

        ids = store.list_ids()
        assert sample_record.person_id in ids
        # corrupted should not appear
        assert "corrupted" not in ids


# ── Test: filename sanitization ──────────────────────────────────────────

class TestFilenameSanitization:
    """Tests for safe filename generation."""

    def test_alphanumeric_unchanged(self):
        assert _sanitize_filename("john_doe_001") == "john_doe_001"

    def test_special_chars_replaced(self):
        assert _sanitize_filename("john@doe.com") == "john_doe_com"

    def test_spaces_replaced(self):
        assert _sanitize_filename("john doe") == "john_doe"

    def test_empty_after_sanitize_raises(self):
        with pytest.raises(ValueError):
            _sanitize_filename("@@@")


# ── Test: is_subclass ────────────────────────────────────────────────────

class TestStorageInterface:
    """Tests for interface compliance."""

    def test_is_subclass(self):
        assert issubclass(LocalEmbeddingStore, EmbeddingStore)
