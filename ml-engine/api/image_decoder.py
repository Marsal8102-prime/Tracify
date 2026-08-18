import cv2
import numpy as np
from fastapi import UploadFile

from api.errors import ErrorCode, MLAPIError

async def decode_upload(file: UploadFile, max_size_bytes: int) -> np.ndarray:
    """
    Validates and decodes an uploaded image file into a BGR numpy array.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise MLAPIError(
            code=ErrorCode.UNSUPPORTED_IMAGE_TYPE,
            message=f"Unsupported image type: {file.content_type}. Only JPEG and PNG are allowed.",
            status_code=415
        )

    # Read the file content
    try:
        contents = await file.read(max_size_bytes + 1)
    except Exception:
        raise MLAPIError(
            code=ErrorCode.INVALID_IMAGE,
            message="Failed to read image file.",
            status_code=400
        )

    if not contents:
        raise MLAPIError(
            code=ErrorCode.INVALID_IMAGE,
            message="Image file is empty.",
            status_code=400
        )

    if len(contents) > max_size_bytes:
        raise MLAPIError(
            code=ErrorCode.IMAGE_TOO_LARGE,
            message=f"Image exceeds the maximum allowed size of {max_size_bytes} bytes.",
            status_code=413
        )

    # Decode image using cv2
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise MLAPIError(
                code=ErrorCode.INVALID_IMAGE,
                message="Image decode failed."
            )

        # Verify it has 3 channels
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise MLAPIError(
                code=ErrorCode.INVALID_IMAGE,
                message="Image must have 3 channels (BGR)."
            )

        return img
    except MLAPIError:
        raise
    except Exception:
        raise MLAPIError(
            code=ErrorCode.INVALID_IMAGE,
            message="Image decode failed."
        )
