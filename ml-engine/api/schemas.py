from typing import List, Optional
from pydantic import BaseModel, Field
from registration.result import RegistrationStatus
from recognition.result import RecognitionStatus

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    gallery_loaded: bool
    gallery_size: int

class SampleResultSchema(BaseModel):
    accepted: bool
    reason: str
    sample_index: int

class RegistrationResponse(BaseModel):
    person_id: str
    status: RegistrationStatus
    accepted_count: int
    rejected_count: int
    rejection_reasons: List[str]
    sample_results: List[SampleResultSchema]
    duplicate_person_id: Optional[str] = None
    duplicate_similarity: Optional[float] = Field(None, ge=-1.0, le=1.0)
    model_name: str
    embedding_dimension: int
    timestamp: str

class FaceResult(BaseModel):
    person_id: Optional[str]
    recognition_status: RecognitionStatus
    similarity: float = Field(..., ge=-1.0, le=1.0)
    threshold: float = Field(..., ge=-1.0, le=1.0)
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[int] = Field(..., min_length=4, max_length=4)
    matched_embedding_id: Optional[str] = None

class RecognitionResponse(BaseModel):
    face_count: int
    processing_time_ms: float
    faces: List[FaceResult]

class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str

class ErrorResponse(BaseModel):
    error: ErrorDetail
