import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI
from contextlib import asynccontextmanager

from backend.app.main import create_app
from backend.app.config import Settings
from backend.app.clients.ml_engine import MLEngineClient


@pytest.mark.asyncio
async def test_polling_task_lifecycle():
    # Setup test app
    app = create_app()

    with patch("backend.app.main.asyncio.create_task") as mock_create_task:
        mock_task = asyncio.Future()
        mock_create_task.return_value = mock_task

        async with app.router.lifespan_context(app):
            # Lifespan yields control here (app is running)

            # Verify the polling task was started exactly once
            mock_create_task.assert_called_once()
            args, _ = mock_create_task.call_args
            assert asyncio.iscoroutine(args[0])
            # prevent 'coroutine not awaited' warning
            args[0].close()

        # Context exit (shutdown) - verify task cancellation
        assert mock_task.cancelled()
