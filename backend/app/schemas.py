from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LivenessResponse(StrictModel):
    status: Literal["ok"]
    version: str


class ReadinessResponse(StrictModel):
    status: Literal["ready"]
    version: str


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorDetail


class MLHealthResponse(StrictModel):
    status: Literal["ok", "unavailable"]
    version: str
    models_loaded: bool
    gallery_loaded: bool
    gallery_size: int = Field(ge=0)


RegistrationStatus = Literal["success", "rejected", "duplicate", "invalid"]


class MLSampleResult(StrictModel):
    accepted: bool
    reason: str
    sample_index: int = Field(ge=0)


class MLRegistrationResponse(StrictModel):
    person_id: str
    status: RegistrationStatus
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    rejection_reasons: list[str]
    sample_results: list[MLSampleResult]
    duplicate_person_id: str | None = None
    duplicate_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    model_name: str
    embedding_dimension: int = Field(gt=0)
    timestamp: str


RecognitionStatus = Literal["known", "unknown"]


class MLFaceResult(StrictModel):
    person_id: str | None
    recognition_status: RecognitionStatus
    similarity: float = Field(ge=-1.0, le=1.0)
    threshold: float = Field(ge=-1.0, le=1.0)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(min_length=4, max_length=4)
    matched_embedding_id: str | None = None


class MLRecognitionResponse(StrictModel):
    face_count: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0)
    faces: list[MLFaceResult]

    @model_validator(mode="after")
    def validate_face_count(self) -> "MLRecognitionResponse":
        if self.face_count != len(self.faces):
            raise ValueError("face_count must equal the number of faces.")
        return self


KnownMLErrorCode = Literal[
    "INVALID_IMAGE",
    "UNSUPPORTED_IMAGE_TYPE",
    "IMAGE_TOO_LARGE",
    "INVALID_METADATA",
    "ML_ENGINE_NOT_READY",
    "ML_PROCESSING_ERROR",
    "VALIDATION_ERROR",
    "INTERNAL_ERROR",
]


class MLDownstreamErrorDetail(StrictModel):
    code: KnownMLErrorCode
    message: str
    request_id: str


class MLDownstreamErrorResponse(StrictModel):
    error: MLDownstreamErrorDetail
