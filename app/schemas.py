"""
Pydantic models for the API request/response contracts.

FastAPI uses these for automatic validation and OpenAPI documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """JSON returned by POST /predict."""

    score: float = Field(..., description="Raw anomaly score (higher = more anomalous).")
    threshold: float = Field(..., description="Calibrated decision threshold.")
    ratio: float = Field(..., description="score / threshold.")
    is_defective: bool = Field(..., description="True if score > threshold.")
    label: str = Field(..., description="'DEFECTIVE' or 'NORMAL'.")
    latency_ms: float = Field(..., description="Server-side model inference time (ms).")

    area_pct: float = Field(
        ..., ge=0, le=100,
        description="Percentage of the image area above the threshold.",
    )
    n_regions: int = Field(
        ..., ge=0,
        description="Number of separate flagged regions (tiny noise regions are ignored).",
    )
    max_location: List[int] = Field(
        ..., min_length=2, max_length=2,
        description="[row, col] of the most anomalous pixel on the 256x256 anomaly map.",
    )

    original_url: Optional[str] = Field(None, description="URL of the analysed image (resized to 256x256).")
    heatmap_url: Optional[str] = Field(None, description="URL of the heatmap overlay PNG.")
    mask_url: Optional[str] = Field(None, description="URL of the predicted defect mask PNG (white = above threshold).")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_source: Optional[str] = Field(None, description="'local' or 'registry'.")
    model_version: Optional[int] = Field(None, description="Registry version (registry mode only).")


class InfoResponse(BaseModel):
    model_source: str
    model_name: Optional[str] = None
    model_alias: Optional[str] = None
    model_version: Optional[int] = None
    threshold: float
    image_size: int
    category: str


class ErrorResponse(BaseModel):
    detail: str