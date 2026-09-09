"""API contract tests: auth, job lifecycle, artifacts, review, audit (TC 08)."""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from src.data.split import read_manifest

CLINICIAN_KEY = "test-clinician-key"
VIEWER_KEY = "test-viewer-key"


@pytest.fixture(scope="module")
def client(checkpoint_path, tmp_path_factory, tiny_cfg):
    os.environ["LIVER3D_API_KEYS"] = f"{CLINICIAN_KEY}:clinician,{VIEWER_KEY}:viewer"
    from src.api.service import create_app

    work_dir = str(tmp_path_factory.mktemp("api"))
    app = create_app(checkpoint=checkpoint_path, work_dir=work_dir, device="cpu")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_volume(tiny_cfg):
    rows = read_manifest(tiny_cfg["data"]["split_manifest"], split="test")
    return rows[0]["image"]


def _submit(client, path, key=CLINICIAN_KEY):
    with open(path, "rb") as fh:
        return client.post(
            "/v1/segmentations",
            files={"file": ("scan.nii.gz", fh, "application/gzip")},
            headers={"X-API-Key": key},
        )


def _wait(client, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/segmentations/{job_id}", headers={"X-API-Key": CLINICIAN_KEY})
        payload = response.json()
        if payload["status"] in ("completed", "failed", "rejected"):
            return payload
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_health_is_public(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_missing_key_is_denied(client, sample_volume):
    with open(sample_volume, "rb") as fh:
        response = client.post("/v1/segmentations", files={"file": ("scan.nii.gz", fh)})
    assert response.status_code == 401
    assert "patient" not in response.text.lower()


def test_viewer_role_cannot_submit(client, sample_volume):
    response = _submit(client, sample_volume, key=VIEWER_KEY)
    assert response.status_code == 403


def test_unsupported_format_is_rejected(client):
    response = client.post(
        "/v1/segmentations",
        files={"file": ("scan.txt", b"not a volume", "text/plain")},
        headers={"X-API-Key": CLINICIAN_KEY},
    )
    assert response.status_code == 400


def test_unknown_job_is_404(client):
    response = client.get("/v1/segmentations/deadbeef", headers={"X-API-Key": CLINICIAN_KEY})
    assert response.status_code == 404


def test_full_job_lifecycle(client, sample_volume):
    accepted = _submit(client, sample_volume)
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    assert accepted.json()["review_required"] is True

    result = _wait(client, job_id)
    assert result["status"] == "completed", result
    assert result["model_version"].startswith("liver-unet-")
    assert result["review_status"] == "pending"
    assert result["mask_url"].endswith("/mask")

    mask = client.get(result["mask_url"], headers={"X-API-Key": CLINICIAN_KEY})
    assert mask.status_code == 200 and len(mask.content) > 0

    review = client.post(
        f"/v1/segmentations/{job_id}/review",
        json={"decision": "corrected", "note": "boundary edited near the dome"},
        headers={"X-API-Key": CLINICIAN_KEY},
    )
    assert review.status_code == 200
    assert review.json()["review_status"] == "corrected"

    denied = client.post(
        f"/v1/segmentations/{job_id}/review",
        json={"decision": "accepted"},
        headers={"X-API-Key": VIEWER_KEY},
    )
    assert denied.status_code == 403

    audit = client.get(f"/v1/segmentations/{job_id}/audit", headers={"X-API-Key": CLINICIAN_KEY})
    actions = [event["action"] for event in audit.json()["events"]]
    assert "job_created" in actions and "job_completed" in actions and "review_recorded" in actions

    listing = client.get("/v1/segmentations", headers={"X-API-Key": VIEWER_KEY})
    assert any(job["job_id"] == job_id for job in listing.json())


def test_model_endpoint_reports_version_and_limits(client):
    response = client.get("/v1/model", headers={"X-API-Key": CLINICIAN_KEY})
    payload = response.json()
    assert payload["model_version"].startswith("liver-unet-")
    assert "not a cleared medical device" in payload["intended_use"]
    assert payload["preprocessing"]["spacing_mm"]


def test_result_reports_measurements_and_finding(client, sample_volume):
    """The browser workstation states a finding from these fields (TC 05)."""
    accepted = _submit(client, sample_volume)
    job_id = accepted.json()["job_id"]
    result = _wait(client, job_id)

    assert result["status"] == "completed"
    measurements = result["measurements"]
    assert measurements["volume_ml"] >= 0
    assert measurements["voxels"] == int(measurements["voxels"])
    assert measurements["threshold"] == 0.5
    assert isinstance(measurements["empty_prediction"], bool)

    finding = result["finding"]
    assert finding
    if measurements["empty_prediction"]:
        assert "No liver was segmented" in finding
        assert "mandatory review" in finding
    else:
        assert finding.startswith("Liver segmented")
        assert "accept, correct or reject" in finding
    assert result["review_required"] is True


def test_cors_preflight_allows_the_local_workstation(client):
    response = client.options(
        "/v1/segmentations",
        headers={
            "Origin": "http://127.0.0.1:8099",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:8099"
