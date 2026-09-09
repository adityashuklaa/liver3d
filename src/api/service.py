"""FastAPI inference service with role-based access, audit and human review.

Run:
    set LIVER3D_CHECKPOINT=outputs/<run_id>/best.pt
    set LIVER3D_API_KEYS=my-secret-key:clinician
    uvicorn src.api.service:app --port 8000

The service returns a proposed segmentation. A qualified reviewer must accept,
correct or reject it; nothing here is a cleared medical device.
"""
from __future__ import annotations

import os
import queue
import secrets
import shutil
import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..data.io import InputValidationError
from ..infer import LiverSegmenter
from ..utils.checksum import file_sha256
from ..utils.logging_utils import get_logger
from .schemas import HealthResponse, JobAccepted, JobResult, ReviewRequest, ReviewResponse
from .store import JobStore

LOG = get_logger("liver3d.api")

ALLOWED_SUFFIXES = (".nii", ".nii.gz")
MAX_UPLOAD_BYTES = int(os.environ.get("LIVER3D_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
# Browser clients (the offline review workstation) are cross-origin, so the
# allowed origins are explicit and configurable. Default covers a local page.
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "LIVER3D_CORS_ORIGINS",
        "http://localhost:8099,http://127.0.0.1:8099,http://localhost:5500,null",
    ).split(",") if o.strip()
]
REVIEW_ROLES = {"clinician", "radiologist", "admin"}
WRITE_ROLES = REVIEW_ROLES | {"technologist"}


def load_api_keys() -> Dict[str, str]:
    """Parse LIVER3D_API_KEYS='key:role,key2:role2'. Generates a dev key if unset."""
    raw = os.environ.get("LIVER3D_API_KEYS", "").strip()
    keys: Dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, _, role = entry.partition(":")
        keys[key.strip()] = (role.strip() or "viewer").lower()
    if not keys:
        dev_key = secrets.token_urlsafe(16)
        keys[dev_key] = "clinician"
        LOG.warning(
            "LIVER3D_API_KEYS is not set. Generated a temporary development key: %s "
            "(set LIVER3D_API_KEYS before any real deployment)",
            dev_key,
        )
    return keys


class Principal:
    def __init__(self, key: str, role: str):
        self.role = role
        # never log or store the key itself
        self.actor = f"{role}:{file_sha256_of_text(key)[:8]}"


def file_sha256_of_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Worker(threading.Thread):
    """Single-threaded job runner: one GPU-sized job at a time."""

    def __init__(self, app_state: "AppState"):
        super().__init__(daemon=True)
        self.state = app_state
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.stop_event = threading.Event()

    def submit(self, job_id: str) -> None:
        self.queue.put(job_id)

    def run(self) -> None:  # pragma: no cover - exercised through the API tests
        while not self.stop_event.is_set():
            try:
                job_id = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.state.process(job_id)
            except Exception as exc:  # defensive: worker must never die
                LOG.exception("job %s failed", job_id)
                self.state.store.update(
                    job_id, status="failed", error_category="internal_error",
                    error_message="inference failed; see server logs",
                    trace_id=uuid.uuid4().hex[:12],
                )
            finally:
                self.queue.task_done()


class AppState:
    def __init__(self, checkpoint: Optional[str], work_dir: str, device: str = "auto"):
        self.checkpoint = checkpoint
        self.work_dir = work_dir
        self.device = device
        self.store = JobStore(os.path.join(work_dir, "jobs.db"))
        self.api_keys = load_api_keys()
        self._segmenter: Optional[LiverSegmenter] = None
        self._lock = threading.Lock()
        self.worker = Worker(self)
        self.worker.start()

    @property
    def segmenter(self) -> LiverSegmenter:
        if self._segmenter is None:
            with self._lock:
                if self._segmenter is None:
                    if not self.checkpoint or not os.path.exists(self.checkpoint):
                        raise HTTPException(
                            status_code=503,
                            detail="model checkpoint not configured; set LIVER3D_CHECKPOINT",
                        )
                    LOG.info("loading checkpoint %s", self.checkpoint)
                    self._segmenter = LiverSegmenter(self.checkpoint, device=self.device)
        return self._segmenter

    def job_dir(self, job_id: str) -> str:
        return os.path.join(self.work_dir, "jobs", job_id)

    def process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        queued_for = time.time() - float(job["created_at"])
        self.store.update(job_id, status="running")
        job_dir = self.job_dir(job_id)
        input_path = os.path.join(job_dir, "input.nii.gz")
        try:
            segmenter = self.segmenter
            audit = segmenter.predict_file(
                input_path, job_dir, job_id=job_id, requester=job.get("requester") or "api"
            )
        except InputValidationError as exc:
            self.store.update(
                job_id, status="rejected", error_category="input_validation",
                error_message=str(exc), trace_id=uuid.uuid4().hex[:12],
            )
            self.store.audit(job_id, "system", "job_rejected", {"reason": str(exc)})
            return
        self.store.update(
            job_id,
            status="completed",
            model_version=audit["model_version"],
            preprocessing_version=str(audit["preprocessing"]),
            runtime_seconds=audit["runtime_seconds"],
            warnings=audit["warnings"],
            result={
                "mask": audit["outputs"]["mask"],
                "probability": audit["outputs"].get("probability"),
                "overlay": audit["outputs"].get("overlay_png"),
                "surface": audit["outputs"].get("surface_html"),
                "geometry": {
                    "shape": audit["input_shape"],
                    "spacing_mm": audit["input_spacing_mm"],
                },
                "measurements": {
                    "volume_ml": audit["postprocess_report"]["volume_ml"],
                    "voxels": audit["postprocess_report"]["voxels_final"],
                    "components_before_largest": audit["postprocess_report"].get("components_before_largest"),
                    "empty_prediction": audit["postprocess_report"]["empty_prediction"],
                    "threshold": audit["inference"]["threshold"],
                },
                "queue_seconds": round(queued_for, 3),
            },
        )
        self.store.audit(job_id, "system", "job_completed", {
            "model_version": audit["model_version"],
            "runtime_seconds": audit["runtime_seconds"],
            "warnings": audit["warnings"],
        })


def create_app(checkpoint: Optional[str] = None, work_dir: Optional[str] = None, device: str = "auto") -> FastAPI:
    checkpoint = checkpoint or os.environ.get("LIVER3D_CHECKPOINT")
    work_dir = work_dir or os.environ.get("LIVER3D_WORK_DIR", "outputs/api")
    os.makedirs(work_dir, exist_ok=True)
    state = AppState(checkpoint, work_dir, device=device)

    app = FastAPI(
        title="Liver 3D U-Net segmentation service",
        version="1.0.0",
        description=(
            "Research prototype. Output is a proposed liver mask that a qualified "
            "reviewer must accept, correct or reject. Not a cleared medical device."
        ),
    )
    app.state.liver = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type"],
        expose_headers=["Content-Disposition"],
    )

    def principal(x_api_key: str = Header(default="", alias="X-API-Key")) -> Principal:
        role = state.api_keys.get(x_api_key)
        if not role:
            state.store.audit(None, "unknown", "access_denied", {"reason": "invalid api key"})
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return Principal(x_api_key, role)

    def require_write(user: Principal = Depends(principal)) -> Principal:
        if user.role not in WRITE_ROLES:
            state.store.audit(None, user.actor, "access_denied", {"reason": "role not permitted"})
            raise HTTPException(status_code=403, detail="role is not permitted to submit studies")
        return user

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        loaded = state._segmenter is not None
        return HealthResponse(
            status="ok",
            model_loaded=loaded,
            model_version=state._segmenter.model_version if loaded else None,
            device=str(state._segmenter.device) if loaded else None,
            queue_depth=state.worker.queue.qsize(),
        )

    @app.get("/v1/model")
    def model_info(user: Principal = Depends(principal)):
        segmenter = state.segmenter
        return {
            "model_version": segmenter.model_version,
            "checkpoint_sha256": segmenter.meta.get("checkpoint_sha256"),
            "preprocessing": {
                "orientation": segmenter.cfg["data"]["orientation"],
                "spacing_mm": segmenter.cfg["data"]["spacing_mm"],
                "intensity_window_hu": segmenter.cfg["data"]["intensity_window_hu"],
            },
            "inference": dict(segmenter.cfg["inference"]),
            "training_meta": {k: v for k, v in segmenter.meta.items() if k != "environment"},
            "intended_use": "research prototype; not a cleared medical device",
        }

    @app.post("/v1/segmentations", response_model=JobAccepted, status_code=202)
    async def create_segmentation(
        file: UploadFile = File(...),
        user: Principal = Depends(require_write),
    ) -> JobAccepted:
        name = (file.filename or "").lower()
        if not name.endswith(ALLOWED_SUFFIXES):
            raise HTTPException(status_code=400, detail="only .nii or .nii.gz uploads are accepted")
        job_id = uuid.uuid4().hex[:16]
        job_dir = state.job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)
        input_path = os.path.join(job_dir, "input.nii.gz")

        size = 0
        with open(input_path, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="upload exceeds the configured size limit")
                fh.write(chunk)
        if size == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="empty upload")

        state.store.create_job(job_id, user.actor, file_sha256(input_path))
        state.worker.submit(job_id)
        return JobAccepted(job_id=job_id, poll=f"/v1/segmentations/{job_id}")

    def _job_or_404(job_id: str) -> dict:
        job = state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        return job

    @app.get("/v1/segmentations", response_model=List[dict])
    def list_jobs(limit: int = 50, user: Principal = Depends(principal)):
        return state.store.list_jobs(limit=limit)

    @app.get("/v1/segmentations/{job_id}", response_model=JobResult)
    def get_result(job_id: str, user: Principal = Depends(principal)) -> JobResult:
        job = _job_or_404(job_id)
        result = job.get("result") or {}
        base = f"/v1/segmentations/{job_id}"
        return JobResult(
            job_id=job_id,
            status=job["status"],
            model_version=job.get("model_version"),
            preprocessing_version=job.get("preprocessing_version"),
            mask_url=f"{base}/mask" if result.get("mask") else None,
            probability_url=f"{base}/probability" if result.get("probability") else None,
            overlay_url=f"{base}/overlay" if result.get("overlay") else None,
            surface_url=f"{base}/surface" if result.get("surface") else None,
            geometry=result.get("geometry"),
            measurements=result.get("measurements"),
            finding=_finding(job, result),
            runtime_seconds=job.get("runtime_seconds"),
            queue_seconds=result.get("queue_seconds"),
            warnings=job.get("warnings") or [],
            review_status=job.get("review_status") or "pending",
            error_category=job.get("error_category"),
            error_message=job.get("error_message"),
            trace_id=job.get("trace_id"),
        )

    def _finding(job: dict, result: dict) -> Optional[str]:
        """One plain sentence a reader can act on. Never a diagnosis."""
        if job["status"] == "rejected":
            return "Input rejected before inference: " + str(job.get("error_message") or "not eligible")
        if job["status"] == "failed":
            return "Inference failed; no clinical output was produced."
        if job["status"] != "completed":
            return None
        measurements = result.get("measurements") or {}
        if measurements.get("empty_prediction"):
            return ("No liver was segmented in this volume. Flagged for mandatory review - "
                    "an empty result is never reported as normal.")
        volume = measurements.get("volume_ml")
        text = "Liver segmented"
        if volume is not None:
            text += f": {volume:.0f} ml as one connected structure"
        text += ". Proposed mask - a qualified reader must accept, correct or reject it."
        return text

    def _artifact(job_id: str, key: str, media_type: str, filename: str) -> FileResponse:
        job = _job_or_404(job_id)
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"job is {job['status']}; no clinical output available")
        path = (job.get("result") or {}).get(key)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"{key} not available for this job")
        return FileResponse(path, media_type=media_type, filename=filename)

    @app.get("/v1/segmentations/{job_id}/mask")
    def get_mask(job_id: str, user: Principal = Depends(principal)):
        state.store.audit(job_id, user.actor, "mask_downloaded")
        return _artifact(job_id, "mask", "application/gzip", f"{job_id}_mask.nii.gz")

    @app.get("/v1/segmentations/{job_id}/probability")
    def get_probability(job_id: str, user: Principal = Depends(principal)):
        return _artifact(job_id, "probability", "application/gzip", f"{job_id}_prob.nii.gz")

    @app.get("/v1/segmentations/{job_id}/overlay")
    def get_overlay(job_id: str, user: Principal = Depends(principal)):
        return _artifact(job_id, "overlay", "image/png", f"{job_id}_overlay.png")

    @app.get("/v1/segmentations/{job_id}/surface")
    def get_surface(job_id: str, user: Principal = Depends(principal)):
        return _artifact(job_id, "surface", "text/html", f"{job_id}_surface.html")

    @app.post("/v1/segmentations/{job_id}/review", response_model=ReviewResponse)
    def review(job_id: str, body: ReviewRequest, user: Principal = Depends(principal)) -> ReviewResponse:
        if user.role not in REVIEW_ROLES:
            raise HTTPException(status_code=403, detail="role is not permitted to review results")
        if body.decision not in ("accepted", "corrected", "rejected"):
            raise HTTPException(status_code=400, detail="decision must be accepted, corrected or rejected")
        _job_or_404(job_id)
        state.store.set_review(job_id, body.decision, user.role, body.note)
        job = state.store.get(job_id)
        return ReviewResponse(
            job_id=job_id,
            review_status=job["review_status"],
            reviewer_role=job["reviewer_role"],
            reviewed_at=job["reviewed_at"],
        )

    @app.get("/v1/segmentations/{job_id}/audit")
    def audit_trail(job_id: str, user: Principal = Depends(principal)):
        _job_or_404(job_id)
        return {"job_id": job_id, "events": state.store.audit_trail(job_id)}

    @app.exception_handler(InputValidationError)
    def _validation_handler(request, exc: InputValidationError):  # pragma: no cover
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
