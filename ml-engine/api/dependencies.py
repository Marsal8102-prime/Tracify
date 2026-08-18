from fastapi import Request, Depends
from api.runtime import MLRuntime
from api.errors import MLAPIError, ErrorCode

def get_runtime(request: Request) -> MLRuntime:
    return request.app.state.ml_runtime

def require_ready_runtime(runtime: MLRuntime = Depends(get_runtime)) -> MLRuntime:
    if not runtime.ready:
        raise MLAPIError(
            code=ErrorCode.ML_ENGINE_NOT_READY,
            message="ML engine is currently unavailable or failed to initialize.",
            status_code=503
        )
    return runtime
