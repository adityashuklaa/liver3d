# Security, privacy and clinical safety

## Use limitation

This is an academic research prototype. It is not a cleared medical device. It
must not be presented as an autonomous diagnostic system or used for patient care
without the required institutional, clinical, quality and regulatory approvals.
Every output is a proposal until a qualified reviewer accepts, corrects or
rejects it.

## Data protection

| Control | Implementation here | Owner action |
|---|---|---|
| Authorized data only | dataset name, license and approval recorded in `docs/data_card.md` | confirm before first training run |
| De-identification | pipeline reads pixel data and geometry only; DICOM tags are not copied into outputs | run a de-identification tool on source data first |
| Anonymous identifiers | API job ids are random hex; case ids come from the manifest | keep patient names out of filenames and case ids |
| No identifiers in logs | logs record run ids, case ids, metrics; API audit stores `role:hash8(key)`, never the key | review log fields after any change |
| Access control | `X-API-Key` header mapped to roles; submit and review are role-gated; denials are audited | issue one key per person or system, rotate on leave |
| Encryption in transit | not terminated by the service | deploy behind TLS (reverse proxy) |
| Encryption at rest | not provided by the service | use encrypted volumes for `data/`, `outputs/`, `LIVER3D_WORK_DIR` |
| Retention | job inputs and outputs stay under `LIVER3D_WORK_DIR` until deleted | schedule deletion; document the period |
| Dependency risk | pinned minimums in `requirements.txt` | run a dependency scan in CI |
| Upload abuse | extension allow-list and `LIVER3D_MAX_UPLOAD_MB` cap | keep the cap at or below available disk |

Never commit patient data, cache files, checkpoints trained on restricted data,
or `.db` job stores. `.gitignore` excludes `data/`, `outputs/` and `*.pt`.

## Clinical safety controls

| Hazard | Control in this implementation |
|---|---|
| Plausible but incorrect mask | overlay PNG + 3D surface for review, probability map export, mandatory review status on every job |
| Empty or implausibly small prediction | flagged in `warnings` and in the audit record; never reported as a normal result |
| Wrong study or series | anonymous job id echoed with input shape, spacing and input checksum for confirmation |
| Geometry mismatch | preprocessing is invertible and verified; mask is exported on the source grid; shape mismatch raises a warning; a test asserts affine equality |
| Input outside validated range | spacing and intensity checks warn or reject before inference |
| Corrupt or unsupported input | rejected with an input eligibility error, job status `rejected`, no mask returned |
| Runtime or memory failure | job marked `failed` with a trace id, no clinical output, error logged for retry |
| Performance drift | `/v1/model` exposes the running version; audit trail records reviewer decisions - monitor correction and rejection rates |
| Overreliance on automation | intended-use statement returned by the API and stored in every audit record |

## Incident response outline

1. Suspend the release (stop the service or revoke API keys).
2. Preserve `LIVER3D_WORK_DIR`, the job database and logs.
3. Identify affected job ids and reviewers from the audit trail.
4. Roll back to the previous release bundle (checkpoint + config + validation report).
5. Record cause, scope, corrective action and re-validation evidence.

## Versioning and rollback

A release is a bundle: checkpoint, its stored config, label definition, inference
protocol, environment record and validation report. `src/checkpoint.py` stores
the config inside the checkpoint, so weights cannot be swapped silently. Rolling
back means restoring the whole bundle, not just the weights.
