# Model card - liver 3D U-Net

Fill every `TO_CONFIRM` before the model is shared, demonstrated as a result, or
used in any report. A card with placeholders is a draft, not a release.

## Model details

| Field | Value |
|---|---|
| Name | liver-unet |
| Version | `liver-unet-<first 12 hex of checkpoint SHA-256>` (from `run_summary.json`) |
| Architecture | 3D U-Net, encoder/bottleneck/decoder with skip connections |
| Feature widths | 32, 64, 128, 256 (config `model.channels`) |
| Blocks | two padded 3x3x3 convolutions per level, instance norm, leaky ReLU |
| Output | one logit channel, sigmoid, threshold from config (`inference.threshold`) |
| Loss | Dice + BCE, smoothing constant 1.0, mean reduction |
| Optimizer | Adam, lr 1e-4, ReduceLROnPlateau on validation Dice |
| Input | single CT intensity channel, RAS, resampled, HU-windowed and scaled to [0,1] |
| Parameters | TO_CONFIRM (printed at training start) |
| Training run | TO_CONFIRM run_id, commit, date |

## Intended use

- Proposing a liver mask on abdominal CT for review by a qualified reader.
- Volumetry, planning support and research, always after human review.

## Out of scope

- Autonomous diagnosis, treatment decisions or clinical clearance.
- Tumour or lesion segmentation (separate labels, separate metrics).
- Modalities other than CT; paediatric or other populations not represented in
  training data; scanners, protocols and contrast phases not evaluated.

## Training data

| Field | Value |
|---|---|
| Dataset and version | TO_CONFIRM |
| License / access conditions | TO_CONFIRM |
| Patients / cases | TO_CONFIRM |
| Split | patient level, manifest `data_manifests/split_v1.csv`, SHA-256 TO_CONFIRM |
| Label definition | liver parenchyma including intrahepatic lesions (LiTS labels 1 and 2 merged) |
| Preprocessing | orientation RAS, spacing TO_CONFIRM mm, HU window TO_CONFIRM |

## Evaluation

| Metric | Validation | Test | 95% CI (test) |
|---|---|---|---|
| Dice | TO_CONFIRM | TO_CONFIRM | TO_CONFIRM |
| IoU | TO_CONFIRM | TO_CONFIRM | TO_CONFIRM |
| Precision | TO_CONFIRM | TO_CONFIRM | TO_CONFIRM |
| Recall | TO_CONFIRM | TO_CONFIRM | TO_CONFIRM |
| HD95 (mm) | TO_CONFIRM | TO_CONFIRM | TO_CONFIRM |
| ASSD (mm) | TO_CONFIRM | TO_CONFIRM | TO_CONFIRM |
| Inference time (s) | TO_CONFIRM | TO_CONFIRM | n/a |

Protocol: metrics per case in source geometry, all eligible cases included,
exclusion rules fixed before looking at results, worst cases reported alongside
the mean. Source: `outputs/<run_id>/eval_test/summary.json`.

## Known failure modes to check and report

- Boundary error next to stomach, heart, right kidney and diaphragm.
- Large lesions, post-surgical anatomy, unusual contrast phase.
- Very thick slices or spacing outside the validated range (flagged as a warning).
- Empty or fragmented prediction - flagged for mandatory review, never treated as
  a normal result.

## Ethical and safety considerations

- Human in the loop: every output is a proposal until a reviewer records accept,
  correct or reject.
- Overreliance risk: intended use and limits are shown by the service (`/v1/model`)
  and belong in any viewer that displays the mask.
- Privacy: see `docs/security_privacy.md`.

## Caveats

Performance measured on one dataset does not transfer automatically to another
site, scanner or population. External validation is required before any claim of
generalization.
