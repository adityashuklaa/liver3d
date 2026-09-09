# API contract

Base URL: `http://<host>:8000`. All endpoints except `/healthz` require
`X-API-Key`. Keys map to roles through `LIVER3D_API_KEYS="key:role,key2:role2"`.

| Role | Submit studies | Read results | Record review |
|---|---|---|---|
| `viewer` | no | yes | no |
| `technologist` | yes | yes | no |
| `clinician`, `radiologist`, `admin` | yes | yes | yes |

The service never stores or logs the key itself; audit rows carry
`role:hash8(key)` as the actor, and job identifiers are random and anonymous.

## Create segmentation job

```
POST /v1/segmentations
Headers: X-API-Key
Body: multipart/form-data, field "file" = .nii or .nii.gz
```

| Response | Meaning |
|---|---|
| `202` | accepted, body `{job_id, status:"queued", review_required:true, poll}` |
| `400` | unsupported format or empty upload |
| `401` | missing or invalid key |
| `403` | role may not submit |
| `413` | upload exceeds `LIVER3D_MAX_UPLOAD_MB` (default 2048) |

Spatially invalid inputs are detected by the worker, not at upload time; the job
then ends with `status:"rejected"` and `error_category:"input_validation"`.

```bash
curl -X POST http://127.0.0.1:8000/v1/segmentations \
  -H "X-API-Key: $KEY" -F "file=@scan.nii.gz"
```

## Get job result

```
GET /v1/segmentations/{job_id}
```

```json
{
  "job_id": "9f1c0f2a5b7d4e10",
  "status": "completed",
  "model_version": "liver-unet-3f2b9c1d4a55",
  "preprocessing_version": "{'orientation': 'RAS', 'spacing_mm': [1.5, 1.5, 2.0], ...}",
  "mask_format": "NIfTI",
  "mask_url": "/v1/segmentations/9f1c0f2a5b7d4e10/mask",
  "probability_url": "/v1/segmentations/9f1c0f2a5b7d4e10/probability",
  "overlay_url": "/v1/segmentations/9f1c0f2a5b7d4e10/overlay",
  "surface_url": "/v1/segmentations/9f1c0f2a5b7d4e10/surface",
  "geometry": {"shape": [512, 512, 130], "spacing_mm": [0.7, 0.7, 2.5]},
  "runtime_seconds": 14.2,
  "queue_seconds": 0.4,
  "warnings": [],
  "review_required": true,
  "review_status": "pending",
  "intended_use": "research prototype; not a cleared medical device"
}
```

- `status`: `queued` | `running` | `completed` | `failed` | `rejected`.
- While pending, no partial clinical output is returned.
- On failure: `error_category`, a safe `error_message`, a `trace_id`, and no mask.
- Artifact endpoints return `409` unless the job is `completed`.

## Artifacts

| Endpoint | Content |
|---|---|
| `GET /v1/segmentations/{job_id}/mask` | binary mask, NIfTI, source geometry |
| `GET /v1/segmentations/{job_id}/probability` | float probability map, NIfTI |
| `GET /v1/segmentations/{job_id}/overlay` | PNG slice grid for review |
| `GET /v1/segmentations/{job_id}/surface` | interactive 3D surface, HTML |

## Record review

```
POST /v1/segmentations/{job_id}/review
{"decision": "accepted" | "corrected" | "rejected", "note": "free text"}
```

Returns `{job_id, review_status, reviewer_role, reviewed_at}`. Only review roles
may call it; the decision is written to the audit trail.

## Audit, listing, model, health

| Endpoint | Purpose |
|---|---|
| `GET /v1/segmentations/{job_id}/audit` | ordered events: created, completed/rejected, downloads, review |
| `GET /v1/segmentations?limit=50` | recent jobs with status and review status |
| `GET /v1/model` | model version, checkpoint SHA-256, preprocessing and inference settings, training meta |
| `GET /healthz` | liveness, whether the model is loaded, queue depth |

## Operational notes

- One worker thread processes jobs serially, so a single GPU is never
  oversubscribed. Scale by running more service instances behind a queue.
- The model is loaded lazily on first use; `/healthz` reports `model_loaded`.
- Job inputs and outputs live under `LIVER3D_WORK_DIR` (default `outputs/api`).
  Apply the retention policy in `docs/security_privacy.md` to that directory.
- Deploy behind TLS. The service does not terminate TLS itself.
