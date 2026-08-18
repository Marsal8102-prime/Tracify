import os

def get_max_image_bytes() -> int:
    """Get max image size from env or default to 10 MiB, capped at 100 MiB."""
    val = os.environ.get("TRACIFY_ML_API_MAX_IMAGE_BYTES", "10485760")
    try:
        limit = int(val)
        if limit <= 0 or limit > 104857600:
            return 10485760
        return limit
    except ValueError:
        return 10485760
