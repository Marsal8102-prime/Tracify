import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch
from api.runtime import initialize_runtime, MLRuntime
from config.settings import Settings


@pytest.fixture
def mock_components():
    """Patch all ML constructors with autospec; use a real Settings() object
    so that accessing nonexistent attributes raises AttributeError."""
    settings = Settings()

    with patch("api.runtime.load_settings", return_value=settings) as m_load, \
         patch("api.runtime.FramePreprocessor", autospec=True) as m_preproc, \
         patch("api.runtime.create_detector", autospec=True) as m_create_det, \
         patch("api.runtime.FaceAligner", autospec=True) as m_aligner, \
         patch("api.runtime.ArcFaceEmbedder", autospec=True) as m_embedder, \
         patch("api.runtime.LocalEmbeddingStore", autospec=True) as m_store, \
         patch("api.runtime.EmbeddingRecognizer", autospec=True) as m_recognizer, \
         patch("api.runtime.RegistrationService", autospec=True) as m_reg:

        yield {
            "load_settings": m_load,
            "FramePreprocessor": m_preproc,
            "create_detector": m_create_det,
            "FaceAligner": m_aligner,
            "ArcFaceEmbedder": m_embedder,
            "LocalEmbeddingStore": m_store,
            "EmbeddingRecognizer": m_recognizer,
            "RegistrationService": m_reg,
            "settings": settings,
        }


def test_initialize_runtime_success(mock_components):
    mocks = mock_components
    settings = mocks["settings"]
    lock = asyncio.Lock()

    runtime = initialize_runtime(settings=None, lock=lock)

    # Verify load_settings called
    mocks["load_settings"].assert_called_once()

    # Verify exact constructor arguments against real Settings attributes
    mocks["FramePreprocessor"].assert_called_once_with(settings.preprocessing)

    mocks["create_detector"].assert_called_once_with(settings.detection)
    mocks["create_detector"].return_value.load_model.assert_called_once()

    mocks["FaceAligner"].assert_called_once_with(settings.alignment)

    mocks["ArcFaceEmbedder"].assert_called_once_with(settings.embedding)
    mocks["ArcFaceEmbedder"].return_value.load_model.assert_called_once()

    mocks["LocalEmbeddingStore"].assert_called_once_with(
        storage_dir=settings.storage.embeddings_dir,
        expected_dimension=settings.embedding.dimension,
    )

    mocks["EmbeddingRecognizer"].assert_called_once_with(
        store=mocks["LocalEmbeddingStore"].return_value,
        config=settings.recognition,
        expected_dimension=settings.embedding.dimension,
    )
    mocks["EmbeddingRecognizer"].return_value.load_gallery.assert_called_once()

    mocks["RegistrationService"].assert_called_once_with(
        detector=mocks["create_detector"].return_value,
        aligner=mocks["FaceAligner"].return_value,
        embedder=mocks["ArcFaceEmbedder"].return_value,
        store=mocks["LocalEmbeddingStore"].return_value,
        recognizer=mocks["EmbeddingRecognizer"].return_value,
        config=settings.registration,
        embedding_dim=settings.embedding.dimension,
    )

    assert runtime.ready is True
    assert runtime.lock is lock
    assert runtime.error is None


def test_initialize_runtime_failure(mock_components):
    mocks = mock_components
    mocks["create_detector"].side_effect = RuntimeError("Model load failed")

    runtime = initialize_runtime()

    assert runtime.ready is False
    assert runtime.error == "Failed to initialize ML models."
    assert runtime.detector is None
    # Verify raw exception text is NOT in the sanitized error
    assert "Model load failed" not in (runtime.error or "")


async def test_lifespan():
    """Run FastAPI lifespan directly; verify startup/shutdown cycle."""
    from api.main import create_app
    from tests.fakes import create_fake_runtime

    def fake_factory(lock=None):
        return create_fake_runtime(ready=False)

    app = create_app(runtime_factory=fake_factory)
    assert not hasattr(app.state, "ml_runtime")

    async with app.router.lifespan_context(app):
        assert hasattr(app.state, "ml_runtime")
        assert app.state.ml_runtime is not None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/internal/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"

    # Cleanup after lifespan exit
    assert app.state.ml_runtime is None


async def test_lifespan_factory_failure():
    """If runtime_factory raises, lifespan installs a ready=False runtime
    so the app stays up and /health returns a sanitized 503."""
    from api.main import create_app

    def failing_factory(lock=None):
        raise RuntimeError("Catastrophic failure")

    app = create_app(runtime_factory=failing_factory)

    async with app.router.lifespan_context(app):
        assert hasattr(app.state, "ml_runtime")
        runtime = app.state.ml_runtime
        assert runtime.ready is False
        assert runtime.error == "ML engine failed to initialize."

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/internal/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["models_loaded"] is False
        assert data["gallery_loaded"] is False
        # No raw exception details leaked
        assert "Catastrophic" not in str(data)
        assert "RuntimeError" not in str(data)

    assert app.state.ml_runtime is None
