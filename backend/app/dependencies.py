from fastapi import Request

from backend.app.clients.ml_engine import MLEngineClient
from backend.app.errors import ServiceUnavailableError


def get_ml_engine_client(request: Request) -> MLEngineClient:
    client = getattr(request.app.state, "ml_engine_client", None)
    if client is None:
        raise ServiceUnavailableError()
    return client
