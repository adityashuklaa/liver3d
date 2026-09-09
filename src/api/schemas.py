"""Request and response models for the segmentation service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"
    review_required: bool = True
    poll: str


class JobResult(BaseModel):
    job_id: str
    status: str = Field(description="queued | running | completed | failed | rejected")
    model_version: Optional[str] = None
    preprocessing_version: Optional[str] = None
    mask_format: str = "NIfTI"
    mask_url: Optional[str] = None
    probability_url: Optional[str] = None
    overlay_url: Optional[str] = None
    surface_url: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    measurements: Optional[Dict[str, Any]] = None
    finding: Optional[str] = None
    runtime_seconds: Optional[float] = None
    queue_seconds: Optional[float] = None
    warnings: List[str] = []
    review_required: bool = True
    review_status: str = "pending"
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    trace_id: Optional[str] = None
    intended_use: str = "research prototype; not a cleared medical device"


class ReviewRequest(BaseModel):
    decision: str = Field(description="accepted | corrected | rejected")
    note: str = ""


class ReviewResponse(BaseModel):
    job_id: str
    review_status: str
    reviewer_role: str
    reviewed_at: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    device: Optional[str] = None
    queue_depth: int = 0
