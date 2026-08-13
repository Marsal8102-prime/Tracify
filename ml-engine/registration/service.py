"""
Registration — Service orchestrator for face registration.

Manages the complete registration lifecycle:
    1. Validate person identity
    2. Process each face sample (detect → align → embed)
    3. Quality-check each sample
    4. Check for duplicate identities
    5. Store validated embeddings
    6. Refresh the recognition gallery
    7. Return structured RegistrationResult

The service does NOT contain low-level detection, alignment, embedding,
or storage logic. It orchestrates existing Phase 1–3 components.

Rollback safety:
    If registration fails after some embeddings have been stored,
    the service cleans up all stored embeddings for this registration
    to avoid leaving the system in an inconsistent state.

Multi-embedding storage strategy:
    Each embedding is stored with a composite key:
        {person_id}__{index}  (double underscore separator)
    The EmbeddingRecord.person_id field retains the real person_id,
    so the recognition engine's _build_candidates() correctly groups
    all embeddings for the same person. The composite key only affects
    the filename in LocalEmbeddingStore.

Usage:
    from registration.service import RegistrationService
    service = RegistrationService(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        store=store,
        recognizer=recognizer,
        config=settings.registration,
        embedding_dim=settings.embedding.dimension,
    )
    result = service.register(
        person_id="EMP-001",
        display_name="Alice Chen",
        face_images=[img1, img2, img3],
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from alignment.base import BaseAligner
from config.settings import RegistrationSettings
from detection.base import BaseDetector
from embedding.base import BaseEmbedder
from recognition.base import BaseRecognizer
from registration.duplicate import DuplicateChecker
from registration.models import PersonIdentity
from registration.quality import FaceQualityValidator
from registration.result import (
    RegistrationResult,
    RegistrationStatus,
    SampleResult,
)
from storage.base import EmbeddingRecord, EmbeddingStore
from utils.logger import get_logger
from utils.timing import timed

_logger = get_logger("tracify.registration.service")

# Separator for composite embedding storage keys.
# Double underscore avoids collision with single-underscore person IDs.
_EMBEDDING_KEY_SEPARATOR = "__emb_"


def _generate_embedding_key(person_id: str, index: int) -> str:
    """Generate a unique storage key for a person's Nth embedding."""
    return f"{person_id}{_EMBEDDING_KEY_SEPARATOR}{index}"


class RegistrationService:
    """
    Orchestrates the complete face registration lifecycle.

    Accepts raw face images, processes them through the ML pipeline,
    validates quality, checks for duplicates, and stores the results.
    """

    def __init__(
        self,
        detector: BaseDetector,
        aligner: BaseAligner,
        embedder: BaseEmbedder,
        store: EmbeddingStore,
        recognizer: BaseRecognizer,
        config: RegistrationSettings,
        embedding_dim: int = 512,
    ):
        """
        Args:
            detector: Face detector (must be loaded).
            aligner: Face aligner.
            embedder: Embedding generator (must be loaded).
            store: Embedding storage backend.
            recognizer: Recognition engine (gallery must be loaded).
            config: Registration-specific settings.
            embedding_dim: Expected embedding dimensionality.
        """
        self._detector = detector
        self._aligner = aligner
        self._embedder = embedder
        self._store = store
        self._recognizer = recognizer
        self._config = config
        self._embedding_dim = embedding_dim

        self._quality = FaceQualityValidator(
            minimum_face_size=config.minimum_face_size,
            embedding_dimension=embedding_dim,
            enabled=config.quality_checks_enabled,
        )
        self._duplicate_checker = DuplicateChecker(recognizer)

        _logger.info(
            f"RegistrationService initialized: "
            f"min_samples={config.minimum_samples}, "
            f"max_samples={config.maximum_samples}, "
            f"duplicate_threshold={config.duplicate_threshold}"
        )

    @timed(name="registration.register")
    def register(
        self,
        person_id: str,
        display_name: str,
        face_images: List[np.ndarray],
        metadata: Optional[Dict[str, str]] = None,
    ) -> RegistrationResult:
        """
        Register a new person with multiple face samples.

        Args:
            person_id: Unique identifier for the person.
            display_name: Human-readable name.
            face_images: List of BGR face images (raw frames).
            metadata: Optional key-value metadata.

        Returns:
            RegistrationResult with status and detailed per-sample results.
        """
        metadata = metadata or {}

        # ── Step 1: Validate person ID ──────────────────────────────
        id_error = self._validate_person_id(person_id)
        if id_error is not None:
            _logger.warning(f"Registration rejected: {id_error}")
            return RegistrationResult(
                person_id=person_id,
                status=RegistrationStatus.INVALID,
                rejection_reasons=[id_error],
            )

        # ── Step 2: Validate sample count ───────────────────────────
        if not face_images:
            return RegistrationResult(
                person_id=person_id,
                status=RegistrationStatus.REJECTED,
                rejection_reasons=["no face images provided"],
            )

        if len(face_images) > self._config.maximum_samples:
            face_images = face_images[: self._config.maximum_samples]
            _logger.info(
                f"Trimmed face samples to maximum "
                f"({self._config.maximum_samples})"
            )

        # ── Step 3: Process each sample ─────────────────────────────
        valid_embeddings: List[np.ndarray] = []
        sample_results: List[SampleResult] = []
        rejection_reasons: List[str] = []

        for idx, image in enumerate(face_images):
            embedding, sample_result = self._process_sample(image, idx)
            sample_results.append(sample_result)

            if sample_result.accepted and embedding is not None:
                valid_embeddings.append(embedding)
            elif not sample_result.accepted:
                if sample_result.reason not in rejection_reasons:
                    rejection_reasons.append(sample_result.reason)

        accepted_count = len(valid_embeddings)
        rejected_count = len(face_images) - accepted_count

        # ── Step 4: Check minimum sample requirement ────────────────
        if accepted_count < self._config.minimum_samples:
            _logger.warning(
                f"Registration rejected for '{person_id}': "
                f"only {accepted_count}/{self._config.minimum_samples} "
                f"valid samples"
            )
            return RegistrationResult(
                person_id=person_id,
                status=RegistrationStatus.REJECTED,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                rejection_reasons=rejection_reasons or [
                    f"insufficient valid samples: "
                    f"{accepted_count}/{self._config.minimum_samples}"
                ],
                sample_results=sample_results,
            )

        # ── Step 5: Duplicate check ─────────────────────────────────
        # Use the first valid embedding as representative for dup check
        dup_result = self._duplicate_checker.check(
            valid_embeddings[0],
            threshold=self._config.duplicate_threshold,
        )
        if dup_result.is_duplicate:
            _logger.warning(
                f"Registration rejected for '{person_id}': "
                f"duplicate of '{dup_result.matched_person_id}' "
                f"(similarity={dup_result.similarity:.4f})"
            )
            return RegistrationResult(
                person_id=person_id,
                status=RegistrationStatus.DUPLICATE,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                sample_results=sample_results,
                duplicate_person_id=dup_result.matched_person_id,
                duplicate_similarity=dup_result.similarity,
            )

        # ── Step 6: Store embeddings ────────────────────────────────
        stored_keys: List[str] = []
        try:
            for idx, embedding in enumerate(valid_embeddings):
                storage_key = _generate_embedding_key(person_id, idx)
                self._store.save(EmbeddingRecord(
                    person_id=storage_key,
                    embedding=embedding,
                    metadata={
                        "display_name": display_name,
                        "embedding_index": str(idx),
                        "model_version": self._embedder.__class__.__name__,
                        **metadata,
                    },
                ))
                stored_keys.append(storage_key)

            # ── Step 7: Refresh gallery ─────────────────────────────
            self._recognizer.refresh_gallery()

        except Exception as e:
            # Rollback: clean up any stored embeddings
            _logger.error(
                f"Registration failed for '{person_id}': {e}. "
                f"Rolling back {len(stored_keys)} stored embeddings."
            )
            self._rollback(stored_keys)
            return RegistrationResult(
                person_id=person_id,
                status=RegistrationStatus.INVALID,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                rejection_reasons=[f"storage failure: {e}"],
                sample_results=sample_results,
            )

        _logger.info(
            f"Registration successful: person_id='{person_id}', "
            f"embeddings_stored={accepted_count}"
        )
        return RegistrationResult(
            person_id=person_id,
            status=RegistrationStatus.SUCCESS,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            rejection_reasons=rejection_reasons,
            sample_results=sample_results,
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _validate_person_id(self, person_id: str) -> Optional[str]:
        """Validate the person ID. Returns error message or None."""
        if not person_id or not person_id.strip():
            return "person_id must be a non-empty string"

        if _EMBEDDING_KEY_SEPARATOR in person_id:
            return (
                f"person_id must not contain the reserved separator "
                f"'{_EMBEDDING_KEY_SEPARATOR}'"
            )

        # Check for existing embeddings with this person_id prefix
        existing_ids = self._store.list_ids()
        for existing_id in existing_ids:
            # Extract the base person_id from composite keys
            base_id = existing_id.split(_EMBEDDING_KEY_SEPARATOR)[0]
            if base_id == person_id:
                return (
                    f"person_id '{person_id}' is already registered "
                    f"(found existing embedding '{existing_id}')"
                )

        return None

    def _process_sample(
        self,
        image: np.ndarray,
        index: int,
    ) -> tuple[Optional[np.ndarray], SampleResult]:
        """
        Process a single face image through the full pipeline.

        Returns:
            (embedding_or_None, SampleResult)
        """
        # Image validation
        ok, reason = self._quality.validate_image(image)
        if not ok:
            return None, SampleResult(
                accepted=False, reason=reason, sample_index=index,
            )

        # Detection
        try:
            detections = self._detector.detect(image)
        except Exception as e:
            return None, SampleResult(
                accepted=False,
                reason=f"detection failed: {e}",
                sample_index=index,
            )

        # Detection quality
        ok, reason = self._quality.validate_detection(detections)
        if not ok:
            return None, SampleResult(
                accepted=False, reason=reason, sample_index=index,
            )

        # Alignment
        try:
            aligned = self._aligner.align(image, detections[0])
        except Exception as e:
            return None, SampleResult(
                accepted=False,
                reason=f"alignment failed: {e}",
                sample_index=index,
            )

        ok, reason = self._quality.validate_aligned(aligned)
        if not ok:
            return None, SampleResult(
                accepted=False, reason=reason, sample_index=index,
            )

        # Embedding generation
        try:
            embedding = self._embedder.generate(aligned)
        except Exception as e:
            return None, SampleResult(
                accepted=False,
                reason=f"embedding generation failed: {e}",
                sample_index=index,
            )

        # Embedding quality
        ok, reason = self._quality.validate_embedding(embedding)
        if not ok:
            return None, SampleResult(
                accepted=False, reason=reason, sample_index=index,
            )

        return embedding, SampleResult(
            accepted=True,
            reason="sample accepted",
            sample_index=index,
        )

    def _rollback(self, stored_keys: List[str]) -> None:
        """Remove stored embeddings on registration failure."""
        for key in stored_keys:
            try:
                deleted = self._store.delete(key)
                if deleted:
                    _logger.debug(f"Rollback: deleted embedding '{key}'")
                else:
                    _logger.warning(
                        f"Rollback: embedding '{key}' not found for deletion"
                    )
            except Exception as e:
                _logger.error(
                    f"Rollback: failed to delete embedding '{key}': {e}"
                )
