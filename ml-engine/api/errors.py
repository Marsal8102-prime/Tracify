import enum

class ErrorCode(str, enum.Enum):
    INVALID_IMAGE = "INVALID_IMAGE"
    UNSUPPORTED_IMAGE_TYPE = "UNSUPPORTED_IMAGE_TYPE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    INVALID_METADATA = "INVALID_METADATA"
    ML_ENGINE_NOT_READY = "ML_ENGINE_NOT_READY"
    ML_PROCESSING_ERROR = "ML_PROCESSING_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"

class MLAPIError(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
