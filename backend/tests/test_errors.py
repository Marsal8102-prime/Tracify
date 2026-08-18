from backend.app.main import create_app
from backend.tests.conftest import request_app


async def test_unexpected_error_is_sanitized(settings, healthy_transport):
    app = create_app(settings=settings)

    @app.get("/explode")
    async def explode():
        raise RuntimeError("secret URL http://internal.example/private C:\\secret")

    response = await request_app(app, "GET", "/explode", headers={"X-Request-ID": "err-1"})
    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An internal server error occurred.",
            "request_id": "err-1",
        }
    }
    assert "internal.example" not in response.text
    assert "C:\\secret" not in response.text
