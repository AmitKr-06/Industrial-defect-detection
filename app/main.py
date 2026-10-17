"""
FastAPI application: /predict, /predict/view, /health, /info, /heatmap.

Two response modes on purpose:
  - POST /predict       -> JSON (for programmatic clients)
  - POST /predict/view  -> HTML (for the browser, uses templates)
"""

import asyncio
import io
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.logger import logging
from src.exception import InvalidInputError

from app import config
from app.dependencies import get_predictor
from app.heatmap_store import save_heatmap, heatmap_path, cleanup_expired
from app.predictor import load_predictor
from app.schemas import (
    PredictionResponse, HealthResponse, InfoResponse, ErrorResponse,
)

CLEANUP_INTERVAL_SECONDS = 600
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")   # image ids: no dots, no slashes


async def _cleanup_loop():
    """Delete expired images every few minutes, for as long as the server runs."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            cleanup_expired()
        except Exception as e:
            logging.warning("Image cleanup failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logging.info("Starting API | MODEL_SOURCE=%s", config.MODEL_SOURCE)
    app.state.predictor = load_predictor()      # loads the model AND warms it up
    cleanup_expired()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    logging.info("API ready")
    yield
    cleanup_task.cancel()
    logging.info("API shutting down")


app = FastAPI(
    title="Industrial Defect Detection API",
    description="Self-supervised PatchCore anomaly detection on MVTec AD.",
    version="1.0.0",
    lifespan=lifespan,
)

# Templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)


# ---------------------------------------------------------------- helpers
async def _read_and_validate(upload: UploadFile) -> bytes:
    """
    Read the uploaded file and validate its content type and size.

    The content type is only a first filter (a client can set it to anything);
    the real check is the image decoding inside the predictor.
    At most MAX_UPLOAD_BYTES + 1 bytes are read, so an oversized upload is
    rejected without loading it into memory.
    """
    if upload.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type {upload.content_type!r}. "
                f"Allowed: PNG, JPG, BMP."
            ),
        )
    data = await upload.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return data


def _predict(predictor, data: bytes) -> dict:
    """
    Run the predictor and prepare its output for the API.

    PIL's Image.open() does not accept raw bytes, so they are wrapped in
    io.BytesIO. Three images (original, heatmap overlay, predicted mask) are
    saved as PNG files and returned as URLs; large arrays are removed from
    the response.
    """
    try:
        result = predictor.predict(io.BytesIO(data))
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))

    overlay = result["overlay"]
    original = Image.open(io.BytesIO(data)).convert("RGB").resize(overlay.size)
    mask_img = Image.fromarray((result["mask"] * 255).astype(np.uint8)).convert("RGB")

    result["original_url"] = f"/heatmap/{save_heatmap(original)}"
    result["heatmap_url"] = f"/heatmap/{save_heatmap(overlay)}"
    result["mask_url"] = f"/heatmap/{save_heatmap(mask_img)}"
    result["max_location"] = list(result["max_location"])

    for key in ("anomaly_map", "mask", "overlay"):
        result.pop(key, None)
    return result


# ---------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Upload page."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", response_model=HealthResponse)
async def health(predictor=Depends(get_predictor)):
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        device=str(getattr(predictor, "device", "cpu")),
        model_source=getattr(predictor, "source", config.MODEL_SOURCE),
        model_version=getattr(predictor, "model_version", None),
    )


@app.get("/info", response_model=InfoResponse)
async def info(predictor=Depends(get_predictor)):
    params = getattr(predictor, "params", {})
    return InfoResponse(
        model_source=config.MODEL_SOURCE,
        model_name=config.MODEL_NAME if config.MODEL_SOURCE == "registry" else None,
        model_alias=config.MODEL_ALIAS if config.MODEL_SOURCE == "registry" else None,
        model_version=getattr(predictor, "model_version", None),
        threshold=float(predictor.threshold),
        image_size=int(params.get("preprocessing", {}).get("image_size", 256)),
        category=params.get("data_ingestion", {}).get("category", "metal_nut"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    predictor=Depends(get_predictor),
):
    """JSON prediction for programmatic clients."""
    data = await _read_and_validate(file)
    return await run_in_threadpool(_predict, predictor, data)   # keeps the event loop free


@app.post("/predict/view", response_class=HTMLResponse)
async def predict_view(
    request: Request,
    file: UploadFile = File(...),
    predictor=Depends(get_predictor),
):
    """
    HTML prediction for the browser.

    On user-level errors (invalid image, too small, wrong type, ...) the upload
    page is shown again with an inline error banner. Unexpected errors show a
    generic message and are written to the log, never raw JSON.
    """
    try:
        data = await _read_and_validate(file)
        result = await run_in_threadpool(_predict, predictor, data)
        return templates.TemplateResponse(request, "result.html", {"result": result})
    except HTTPException as e:
        return templates.TemplateResponse(
            request, "index.html", {"error": e.detail}, status_code=e.status_code,
        )
    except Exception as e:
        logging.error("Unexpected error in /predict/view: %s", e, exc_info=True)
        return templates.TemplateResponse(
            request, "index.html",
            {"error": "Something went wrong while analysing the image. Please try again."},
            status_code=500,
        )


@app.get("/heatmap/{file_id}")
async def get_heatmap(file_id: str):
    """Serve a previously generated PNG (original, overlay or mask)."""
    if not SAFE_ID.fullmatch(file_id):
        raise HTTPException(status_code=404, detail="Image not found or expired.")
    path = heatmap_path(file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found or expired.")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------- error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=True)