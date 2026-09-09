# 3D Liver Segmentation from Volumetric CT Scans (3D U-Net)

Working implementation of the pipeline described in the project documentation:
authorized CT volume in, reviewable liver mask out, with the reproducibility and
safety controls the report calls for.

**Intended use:** research prototype. Output is a *proposed* segmentation that a
qualified reviewer must accept, correct or reject. It is not a cleared medical
device and must not be used for patient care without institutional, clinical,
quality and regulatory approval.

---

## What is in here

| Stage | Entry point | Output |
|---|---|---|
| Synthetic data (no patient data needed) | `scripts/make_synthetic_dataset.py` | NIfTI phantoms |
| Patient-level split | `scripts/build_manifest.py` | `data_manifests/*.csv` + meta |
| Training | `python -m src.train` | checkpoint, logs, curves, config, env |
| Locked evaluation | `python -m src.evaluate` | per-case CSV, aggregate + CI, overlays |
| Single-volume inference | `python -m src.infer` | mask + probability + audit JSON |
| Service | `uvicorn src.api.service:app` | REST API with roles, audit, review |
| Tests | `pytest` | 45 unit / integration / contract tests |
| Review workstation | `scripts/export_viewer_data.py` -> `viewer.html` | offline reviewer UI over the exported masks |

Pipeline: `load -> validate -> orient RAS -> resample -> HU window -> 3D U-Net
patches -> sliding-window inference -> postprocess -> invert to source geometry
-> overlay + 3D surface -> human review`.

## Install

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
# python3 -m venv .venv && source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
# GPU: install the CUDA build of torch first, see https://pytorch.org
```

## 60-second proof it works (no dataset required)

```bash
python scripts/quickstart.py
```

Generates 8 synthetic abdominal phantoms, locks a split, trains a small 3D U-Net
on CPU, evaluates the held-out split, and segments one volume. Artifacts land in
`outputs/demo/`. Synthetic scores are a wiring check, not a clinical result.

## Real dataset (LiTS / CHAOS / your own)

1. Put the data on disk in one of the supported layouts:
   - `imagesTr/case.nii.gz` + `labelsTr/case.nii.gz`, or
   - `images/` + `labels/`, or
   - flat LiTS style `volume-0.nii` + `segmentation-0.nii`.
2. Record the dataset name, version, license and access conditions in
   `configs/baseline.yaml` (`data.dataset`) and `docs/data_card.md`.
3. Build and lock the split (patient level, seeded, checksummed):

```bash
python scripts/build_manifest.py --root data/lits --out data_manifests/split_v1.csv \
    --ratios 0.7 0.15 0.15 --seed 42
# several studies per patient? add e.g. --patient-pattern "(patient\d+)_study\d+"
```

4. Train, then evaluate once on the test split when development is finished:

```bash
python -m src.train --config configs/baseline.yaml
python -m src.evaluate --checkpoint outputs/<run_id>/best.pt --split val    # tuning
python -m src.evaluate --checkpoint outputs/<run_id>/best.pt --split test   # locked
```

5. Segment a new volume (NIfTI file or DICOM series directory):

```bash
python -m src.infer --checkpoint outputs/<run_id>/best.pt --input scan.nii.gz \
    --out-dir outputs/inference
```

Writes `<job>_mask.nii.gz` (source geometry), `<job>_prob.nii.gz`,
`<job>_overlay.png`, `<job>_surface.html` and `<job>_audit.json`.

## Review workstation (single HTML file)

```bash
python scripts/export_viewer_data.py --checkpoint outputs/demo/best.pt --out outputs/viewer
```

Exports slice sprites, masks, per-slice Dice, metrics and a decimated surface for
every case, then `viewer.html` reads them: worklist with quality bands, slice
scrubbing with cine, five view modes (overlay, prediction, reference, difference,
CT only), per-slice plot, rotatable 3D mask, training curves, service transcript,
provenance and a reader accept / correct / reject action. One file, no server,
no CDN - open it or email it.

**Upload a study from the viewer.** The *Upload study* tab sends a volume to the
running service and loads the returned mask back into the review pane with a
stated finding ("Liver segmented: 232 ml as one connected structure"), volume,
slice count, job id and warnings. The page parses NIfTI itself for display; the
segmentation runs in the service on the real model. Start the service first and
allow the page's origin:

```bash
set LIVER3D_CHECKPOINT=outputs/demo/best.pt
set LIVER3D_API_KEYS=demo-key:clinician
set LIVER3D_CORS_ORIGINS=http://127.0.0.1:8099,null
uvicorn src.api.service:app --port 8000
```

Opening `viewer.html` straight from disk works (origin `null`, already allowed).
A page served over HTTPS cannot call a local HTTP service - the viewer says so
instead of failing silently.

## Service

```bash
set LIVER3D_CHECKPOINT=outputs/<run_id>/best.pt
set LIVER3D_API_KEYS=some-long-random-key:clinician
uvicorn src.api.service:app --host 127.0.0.1 --port 8000
```

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/segmentations` | upload `.nii/.nii.gz`, returns `202` + `job_id` |
| GET | `/v1/segmentations/{job_id}` | status, warnings, artifact URLs, review status |
| GET | `/v1/segmentations/{job_id}/mask` | NIfTI mask in source geometry |
| GET | `/v1/segmentations/{job_id}/probability` | probability map |
| GET | `/v1/segmentations/{job_id}/overlay` `/surface` | review artifacts |
| POST | `/v1/segmentations/{job_id}/review` | `accepted` / `corrected` / `rejected` |
| GET | `/v1/segmentations/{job_id}/audit` | audit trail for the job |
| GET | `/v1/model` | model version, checkpoint checksum, preprocessing, limits |
| GET | `/healthz` | liveness, queue depth |

Auth is an `X-API-Key` header mapped to a role (`viewer` read-only, `technologist`
submit, `clinician`/`radiologist`/`admin` submit + review). With no
`LIVER3D_API_KEYS` set, a random development key is generated and logged - set the
variable before any real deployment. Full contract: [docs/api.md](docs/api.md).

Docker:

```bash
docker build -t liver3d .
docker run -p 8000:8000 -e LIVER3D_API_KEYS=key:clinician \
    -v %cd%/outputs:/app/outputs liver3d
```

## Reproducibility

Every run writes `outputs/<run_id>/`: `config.yaml`, `env.json` (versions, commit,
GPU), a copy of the split manifest plus its SHA-256, `train_log.jsonl`,
`metrics.csv`, `curves.png`, `best.pt`, `last.pt`, `run_summary.json` with the
checkpoint checksum and derived `model_version`. Checkpoints carry their own
config, so a release is weights + preprocessing together; loading a checkpoint
uses its stored config unless you deliberately override it.

## Evaluation

Metrics are computed **per case in the source voxel grid** after inverting
preprocessing - Dice, IoU, precision, recall, HD95 (mm), ASSD (mm), predicted and
reference volume in ml, component count, inference seconds. Aggregates report
mean, median, sd, IQR and a bootstrap 95% CI, plus the three worst cases. Fill in
`docs/validation_report_template.md` before quoting any number.

## Status of performance claims

No dataset, split, checkpoint or metric export was supplied with the source
documents, so the reported "Dice above 0.90" is **not reproduced or verified
here**. The results table stays pending until this pipeline is run on a named
dataset with a locked split. See `docs/validation_report_template.md`.

## Repository layout

```
configs/           versioned run configuration (baseline + CPU demo)
data_manifests/    locked patient-level splits and split metadata
scripts/           synthetic data, manifest builder, quickstart
src/config.py      config loading, merging, hashing
src/data/          io, preprocessing (invertible), validation, split, dataset, phantoms
src/models/        3D U-Net
src/losses.py      Dice / BCE / Dice+BCE
src/metrics.py     overlap + surface metrics, bootstrap aggregation
src/inference.py   sliding-window inference with Gaussian blending
src/postprocess.py threshold, component rules, hole filling
src/train.py       training entry point
src/evaluate.py    locked evaluation entry point
src/infer.py       single-volume inference + audit record
src/visualize.py   slice overlays, interactive 3D surface
src/api/           FastAPI service, SQLite job/audit store
tests/             unit, integration and API contract tests
docs/              model card, data card, validation report, safety, API
```

## Tests

```bash
pytest -q            # all tests, ~15 s on CPU
pytest -q -k api     # service contract only
```

Covered: geometry round-trip, geometry mismatch rejection, split leakage,
foreground sampling, model shapes, loss behaviour, metric values, sliding-window
blending, postprocessing rules, empty-prediction flagging, train/evaluate/infer
end to end, source-geometry restoration, and API auth/lifecycle/review/audit.

## Limits

- Liver vs background only. Tumour segmentation is a separate task with separate
  labels and metrics.
- Public-dataset performance does not transfer automatically to a new hospital,
  scanner, protocol or population.
- A high Dice can still hide a clinically important boundary error - review the
  worst cases, not just the mean.
- Deployment components here are a research blueprint, not completed regulatory
  validation.
