import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable, List, Optional

from fastapi import FastAPI, Request, Response, UploadFile, File, Form, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.config import get_max_image_bytes
from api.dependencies import get_runtime, require_ready_runtime
from api.errors import MLAPIError, ErrorCode
from api.image_decoder import decode_upload
from api.runtime import MLRuntime, initialize_runtime
from api.schemas import (
    HealthResponse,
    RegistrationResponse,
    SampleResultSchema,
    RecognitionResponse,
    FaceResult,
    ErrorResponse,
)

logger = logging.getLogger("tracify.api.main")

VERSION = "0.1.0"
REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")

def create_app(runtime_factory: Callable[..., MLRuntime] = initialize_runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lock = asyncio.Lock()
        try:
            app.state.ml_runtime = runtime_factory(lock=lock)
        except Exception:
            logger.error("Runtime factory failed during startup")
            app.state.ml_runtime = MLRuntime(
                settings=None,
                preprocessor=None,
                detector=None,
                aligner=None,
                embedder=None,
                store=None,
                recognizer=None,
                registration_service=None,
                lock=lock,
                ready=False,
                error="ML engine failed to initialize.",
            )
        yield
        # Cleanup
        app.state.ml_runtime = None

    app = FastAPI(title="Tracify ML Engine Internal API", version=VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", "").strip()
        if not req_id or not REQUEST_ID_REGEX.match(req_id):
            req_id = str(uuid.uuid4())

        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "The request data is invalid.",
                    "request_id": req_id
                }
            }
        )

    @app.exception_handler(MLAPIError)
    async def ml_api_exception_handler(request: Request, exc: MLAPIError):
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code.value,
                    "message": exc.message,
                    "request_id": req_id
                }
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, _exc: Exception):
        logger.error("Unhandled API exception")
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": "An internal server error occurred.",
                    "request_id": req_id
                }
            }
        )

    @app.get(
        "/internal/v1/health",
        response_model=HealthResponse,
        responses={
            503: {"model": HealthResponse, "description": "ML engine unavailable"},
        },
    )
    async def health(response: Response, runtime: MLRuntime = Depends(get_runtime)):
        if not runtime.ready:
            response.status_code = 503
            return HealthResponse(
                status="unavailable",
                version=VERSION,
                models_loaded=False,
                gallery_loaded=False,
                gallery_size=0
            )

        return HealthResponse(
            status="ok",
            version=VERSION,
            models_loaded=runtime.embedder.is_loaded and runtime.detector.is_loaded,
            gallery_loaded=runtime.ready,
            gallery_size=runtime.recognizer.gallery_size
        )

    @app.post(
        "/internal/v1/faces/register",
        response_model=RegistrationResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Invalid image or metadata"},
            413: {"model": ErrorResponse, "description": "Image too large"},
            415: {"model": ErrorResponse, "description": "Unsupported image type"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
            503: {"model": ErrorResponse, "description": "ML engine unavailable"},
        },
    )
    async def register_face(
        person_id: str = Form(...),
        display_name: str = Form(...),
        metadata: Optional[str] = Form(None),
        images: List[UploadFile] = File(...),
        runtime: MLRuntime = Depends(require_ready_runtime)
    ):
        if not images:
            raise MLAPIError(ErrorCode.VALIDATION_ERROR, "No images provided.")

        if len(images) > runtime.settings.registration.maximum_samples:
            raise MLAPIError(
                ErrorCode.VALIDATION_ERROR,
                f"Maximum of {runtime.settings.registration.maximum_samples} images allowed.",
                status_code=422
            )

        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                if not isinstance(parsed_metadata, dict):
                    raise ValueError
                if len(parsed_metadata) > 50:
                    raise MLAPIError(ErrorCode.INVALID_METADATA, "Metadata contains too many keys.")
                for k, v in parsed_metadata.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        raise ValueError
                    if len(k) > 100 or len(v) > 1000:
                        raise MLAPIError(ErrorCode.INVALID_METADATA, "Metadata key or value is too long.")
            except ValueError:
                raise MLAPIError(ErrorCode.INVALID_METADATA, "Metadata must be a valid JSON object with string keys and string values.")

        max_size = get_max_image_bytes()
        decoded_images = []
        for img in images:
            decoded = await decode_upload(img, max_size)
            decoded_images.append(decoded)

        # Run registration in thread with lock
        async with runtime.lock:
            try:
                result = await asyncio.to_thread(
                    runtime.registration_service.register,
                    person_id=person_id,
                    display_name=display_name,
                    face_images=decoded_images,
                    metadata=parsed_metadata
                )
            except Exception:
                logger.exception("Registration processing failed")
                raise MLAPIError(ErrorCode.ML_PROCESSING_ERROR, "An error occurred during registration processing.", 500)

        # Map to response schema
        sample_results = [
            SampleResultSchema(
                accepted=s.accepted,
                reason=s.reason,
                sample_index=s.sample_index
            ) for s in result.sample_results
        ]

        return RegistrationResponse(
            person_id=result.person_id,
            status=result.status.value,
            accepted_count=result.accepted_count,
            rejected_count=result.rejected_count,
            rejection_reasons=result.rejection_reasons,
            sample_results=sample_results,
            duplicate_person_id=result.duplicate_person_id,
            duplicate_similarity=result.duplicate_similarity,
            model_name=runtime.settings.embedding.model_name,
            embedding_dimension=runtime.settings.embedding.dimension,
            timestamp=result.timestamp
        )

    @app.post(
        "/internal/v1/faces/recognize",
        response_model=RecognitionResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Invalid image"},
            413: {"model": ErrorResponse, "description": "Image too large"},
            415: {"model": ErrorResponse, "description": "Unsupported image type"},
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
            503: {"model": ErrorResponse, "description": "ML engine unavailable"},
        },
    )
    async def recognize_face(
        image: UploadFile = File(...),
        runtime: MLRuntime = Depends(require_ready_runtime)
    ):
        max_size = get_max_image_bytes()
        decoded_img = await decode_upload(image, max_size)

        def sync_recognize():
            start_time = time.perf_counter()
            processed = runtime.preprocessor.process(decoded_img)
            detections = runtime.detector.detect(processed.frame)

            face_results = []
            for det in detections:
                aligned = runtime.aligner.align(processed.frame, det)
                if aligned is None:
                    continue # Skip faces that can't be aligned

                embedding = runtime.embedder.generate(aligned)
                rec_result = runtime.recognizer.recognize(embedding)

                # Scale detection back to original image
                orig_det = det.scale_to_original(processed.scale_factor)

                face_results.append(
                    FaceResult(
                        person_id=rec_result.person_id,
                        recognition_status=rec_result.status.value,
                        similarity=rec_result.similarity,
                        threshold=rec_result.threshold,
                        detection_confidence=float(det.confidence),
                        bbox=[
                            int(orig_det.bbox[0]),
                            int(orig_det.bbox[1]),
                            int(orig_det.bbox[2]),
                            int(orig_det.bbox[3])
                        ],
                        matched_embedding_id=rec_result.matched_embedding_id
                    )
                )

            end_time = time.perf_counter()
            return face_results, (end_time - start_time) * 1000

        async with runtime.lock:
            try:
                face_results, process_time = await asyncio.to_thread(sync_recognize)
            except Exception:
                logger.exception("Recognition processing failed")
                raise MLAPIError(ErrorCode.ML_PROCESSING_ERROR, "An error occurred during recognition processing.", 500)

        return RecognitionResponse(
            face_count=len(face_results),
            processing_time_ms=process_time,
            faces=face_results
        )

    return app

app = create_app()
