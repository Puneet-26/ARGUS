from __future__ import annotations

import json
import math
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

app = FastAPI(
    title="ARGUS OpenCV Processing API",
    version="0.1.0",
    description="Real CPU-based OpenCV operations for the ARGUS Level 1 workshop.",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 6000
MAX_IMAGE_PIXELS = 24_000_000
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

OPERATIONS = [
    {
        "id": "grayscale",
        "label": "Grayscale",
        "description": "Convert the color image to a single luminance channel.",
        "cvFunction": "cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)",
    },
    {
        "id": "brightness_contrast",
        "label": "Brightness + contrast",
        "description": "Scale pixel intensity and offset brightness.",
        "cvFunction": "cv2.convertScaleAbs()",
    },
    {
        "id": "equalize_hist",
        "label": "Histogram equalization",
        "description": "Redistribute grayscale intensities across the available range.",
        "cvFunction": "cv2.equalizeHist()",
    },
    {
        "id": "clahe",
        "label": "CLAHE",
        "description": "Apply contrast-limited adaptive histogram equalization.",
        "cvFunction": "cv2.createCLAHE().apply()",
    },
    {
        "id": "gaussian_blur",
        "label": "Gaussian blur",
        "description": "Smooth local detail with a Gaussian kernel.",
        "cvFunction": "cv2.GaussianBlur()",
    },
    {
        "id": "threshold",
        "label": "Binary threshold",
        "description": "Set grayscale pixels above a cutoff to white and the rest to black.",
        "cvFunction": "cv2.threshold()",
    },
    {
        "id": "adaptive_threshold",
        "label": "Adaptive threshold",
        "description": "Choose a threshold from a local neighborhood around each pixel.",
        "cvFunction": "cv2.adaptiveThreshold()",
    },
    {
        "id": "resize",
        "label": "Resize",
        "description": "Resample the image to a chosen scale.",
        "cvFunction": "cv2.resize()",
    },
    {
        "id": "crop",
        "label": "Crop region",
        "description": "Select a rectangular region using NumPy array slicing.",
        "cvFunction": "NumPy slicing",
    },
]
OPERATION_IDS = {operation["id"] for operation in OPERATIONS}
OPERATION_PARAMETERS = {
    "grayscale": set(),
    "brightness_contrast": {"contrast", "brightness"},
    "equalize_hist": set(),
    "clahe": {"clip_limit", "tile_grid_size"},
    "gaussian_blur": {"kernel_size"},
    "threshold": {"threshold"},
    "adaptive_threshold": {"block_size", "constant"},
    "resize": {"scale"},
    "crop": {"x", "y", "width", "height"},
}


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Any, _exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "The upload or processing parameters are malformed."},
    )


@app.get("/healthz")
async def healthz() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "pythonAvailable": True,
        "opencvVersion": cv2.__version__,
    }


@app.get("/operations")
async def list_operations() -> list[dict[str, str]]:
    return OPERATIONS


@app.post("/image/inspect")
async def inspect_image(image: UploadFile = File(...)) -> dict[str, Any]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, WebP, or BMP image.",
        )

    raw_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 10 MB or smaller.")

    try:
        decoded = cv2.imdecode(
            np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
    except cv2.error as exc:
        raise HTTPException(
            status_code=400, detail="OpenCV could not decode this image."
        ) from exc
    if decoded is None or decoded.size == 0:
        raise HTTPException(status_code=400, detail="OpenCV could not decode this image.")
    if decoded.ndim not in (2, 3) or (
        decoded.ndim == 3 and decoded.shape[2] not in (1, 3, 4)
    ):
        raise HTTPException(
            status_code=400, detail="The image has an unsupported channel layout."
        )

    height, width = decoded.shape[:2]
    if (
        width > MAX_IMAGE_DIMENSION
        or height > MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Images must be no larger than {MAX_IMAGE_DIMENSION}px per side and "
            f"{MAX_IMAGE_PIXELS // 1_000_000} megapixels.",
        )

    channels = 1 if decoded.ndim == 2 else int(decoded.shape[2])
    color_model = {1: "GRAY", 3: "BGR", 4: "BGRA"}.get(channels)
    if color_model is None:
        raise HTTPException(status_code=400, detail="The image channel layout is unsupported.")

    return {
        "width": int(width),
        "height": int(height),
        "channels": channels,
        "colorModel": color_model,
        "bitDepth": int(decoded.dtype.itemsize * 8),
        "mimeType": image.content_type,
    }


def _finite_number(
    parameters: dict[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    value = parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=400, detail=f"{key} must be a number.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be between {minimum:g} and {maximum:g}.",
        )
    return result


def _integer(
    parameters: dict[str, Any], key: str, default: int, minimum: int, maximum: int
) -> int:
    value = parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(status_code=400, detail=f"{key} must be a whole number.")
    if value < minimum or value > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be between {minimum} and {maximum}.",
        )
    return value


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _apply_operation(
    image: np.ndarray, operation: str, parameters: dict[str, Any]
) -> np.ndarray:
    if operation == "grayscale":
        return _gray(image)

    if operation == "brightness_contrast":
        alpha = _finite_number(parameters, "contrast", 1.0, 0.0, 3.0)
        beta = _finite_number(parameters, "brightness", 0.0, -100.0, 100.0)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if operation == "equalize_hist":
        return cv2.equalizeHist(_gray(image))

    if operation == "clahe":
        clip_limit = _finite_number(parameters, "clip_limit", 2.0, 1.0, 8.0)
        tile_grid_size = _integer(parameters, "tile_grid_size", 8, 2, 32)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        return clahe.apply(_gray(image))

    if operation == "gaussian_blur":
        kernel_size = _integer(parameters, "kernel_size", 5, 1, 51)
        if kernel_size % 2 == 0:
            raise HTTPException(
                status_code=400,
                detail="kernel_size must be odd.",
            )
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    if operation == "threshold":
        threshold_value = _integer(parameters, "threshold", 127, 0, 255)
        _threshold, result = cv2.threshold(
            _gray(image), threshold_value, 255, cv2.THRESH_BINARY
        )
        return result

    if operation == "adaptive_threshold":
        block_size = _integer(parameters, "block_size", 11, 3, 99)
        if block_size % 2 == 0:
            raise HTTPException(status_code=400, detail="block_size must be odd.")
        constant = _finite_number(parameters, "constant", 2.0, -20.0, 20.0)
        return cv2.adaptiveThreshold(
            _gray(image),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant,
        )

    if operation == "resize":
        scale = _finite_number(parameters, "scale", 0.75, 0.1, 2.5)
        height, width = image.shape[:2]
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        if (
            new_width > MAX_IMAGE_DIMENSION
            or new_height > MAX_IMAGE_DIMENSION
            or new_width * new_height > MAX_IMAGE_PIXELS
        ):
            raise HTTPException(
                status_code=400,
                detail="The selected scale would create an image that is too large.",
            )
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    if operation == "crop":
        image_height, image_width = image.shape[:2]
        x = _integer(parameters, "x", 0, 0, image_width - 1)
        y = _integer(parameters, "y", 0, 0, image_height - 1)
        width = _integer(parameters, "width", image_width, 1, image_width)
        height = _integer(parameters, "height", image_height, 1, image_height)
        if x + width > image_width or y + height > image_height:
            raise HTTPException(
                status_code=400,
                detail="The crop rectangle must fit inside the source image.",
            )
        return image[y : y + height, x : x + width].copy()

    raise HTTPException(status_code=400, detail="Select a supported OpenCV operation.")


@app.post("/image/process")
async def process_image(
    image: UploadFile = File(...),
    operation: str = Form(...),
    parameters: str = Form("{}"),
) -> Response:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, WebP, or BMP image.",
        )
    if operation not in OPERATION_IDS:
        raise HTTPException(status_code=400, detail="Select a supported OpenCV operation.")

    raw_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 10 MB or smaller.")

    try:
        decoded = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise HTTPException(status_code=400, detail="OpenCV could not decode this image.") from exc
    if decoded is None or decoded.size == 0:
        raise HTTPException(status_code=400, detail="OpenCV could not decode this image.")
    if decoded.ndim not in (2, 3) or (decoded.ndim == 3 and decoded.shape[2] not in (1, 3, 4)):
        raise HTTPException(status_code=400, detail="The image has an unsupported channel layout.")

    height, width = decoded.shape[:2]
    if (
        width > MAX_IMAGE_DIMENSION
        or height > MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Images must be no larger than {MAX_IMAGE_DIMENSION}px per side and "
            f"{MAX_IMAGE_PIXELS // 1_000_000} megapixels.",
        )
    if decoded.ndim == 3 and decoded.shape[2] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 1:
        decoded = decoded[:, :, 0]

    try:
        parsed_parameters = json.loads(parameters)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Parameters must be valid JSON.") from exc
    if not isinstance(parsed_parameters, dict):
        raise HTTPException(status_code=400, detail="Parameters must be a JSON object.")
    unknown_parameters = set(parsed_parameters) - OPERATION_PARAMETERS[operation]
    if unknown_parameters:
        raise HTTPException(
            status_code=400,
            detail="One or more parameters do not apply to the selected operation.",
        )

    try:
        result = _apply_operation(decoded, operation, parsed_parameters)
        encoded, png = cv2.imencode(".png", result)
    except cv2.error as exc:
        raise HTTPException(status_code=400, detail="OpenCV could not process this image.") from exc
    if not encoded:
        raise HTTPException(status_code=500, detail="OpenCV could not encode the result.")

    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={"X-OpenCV-Operation": operation, "Cache-Control": "no-store"},
    )
