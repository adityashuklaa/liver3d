# Validation report (template)

Complete this after a locked test-set evaluation. Until it is complete, the
project status is "reported performance pending verification" - no Dice number
should be quoted in a report, paper or presentation.

## 1. Release under test

| Field | Value |
|---|---|
| Run id | |
| Code commit | (`outputs/<run_id>/env.json` -> `code_commit`) |
| Config | `outputs/<run_id>/config.yaml` |
| Checkpoint | `outputs/<run_id>/best.pt` |
| Checkpoint SHA-256 | (`run_summary.json`) |
| Model version | `liver-unet-...` |
| Split manifest + SHA-256 | |
| Environment | python / torch / CUDA / GPU from `env.json` |
| Evaluated on | date, operator |

## 2. Data

| Field | Value |
|---|---|
| Dataset and version | |
| Cases: train / val / test | |
| Patients: train / val / test | |
| Excluded cases and rule | |
| Exclusion rule fixed before results were seen? | yes / no |

## 3. Results (from `eval_test/summary.json`)

| Metric | Validation mean | Test mean | Test median | Test 95% CI | n |
|---|---|---|---|---|---|
| Dice | | | | | |
| IoU | | | | | |
| Precision | | | | | |
| Recall | | | | | |
| HD95 (mm) | | | | | |
| ASSD (mm) | | | | | |
| Volume error (ml) | | | | | |
| Inference time (s) | | | | | n/a |

Hardware for timing: ____ . Warm-up runs excluded: ____ .

## 4. Worst cases

| Case | Dice | HD95 | What went wrong | Clinically relevant? |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

## 5. Required figures

- [ ] Training and validation loss by epoch (`curves.png`)
- [ ] Validation Dice by epoch with the selected checkpoint marked (`curves.png`)
- [ ] Best / median / worst overlays (`eval_test/overlays/`)
- [ ] Per-case Dice distribution and failure categories
- [ ] Runtime and memory by input size or patch setting

## 6. Acceptance gates

| Gate | Evidence | Pass? |
|---|---|---|
| Technical validity: no leakage, metrics reproduce from the locked checkpoint and manifest | | |
| Segmentation quality: pre-agreed overlap and surface thresholds + worst-case analysis | | |
| Spatial validity: exported mask overlays correctly in source geometry for all test cases | | |
| Operational performance: measured runtime and memory meet the target environment | | |
| Clinical review: qualified reviewer scored a defined sample (accept / correct / reject) | | |
| Safety: invalid inputs rejected or flagged; no silent plausible-looking failure | | |

Thresholds must be agreed with the guide and clinical reviewer **before** the
test set is opened. A Dice target alone is not clinical acceptance.

## 7. Reproduction check

```bash
python -m src.evaluate --checkpoint <checkpoint> --split test
# compare summary.json with section 3; note any difference and its cause
```

| Item | Result |
|---|---|
| Re-run reproduces the table | yes / no |
| Difference observed | |
| Explanation (nondeterministic kernels, environment change, ...) | |

## 8. Sign-off

| Role | Name | Date | Decision |
|---|---|---|---|
| Project manager | | | |
| PBL guide | | | |
| Clinical reviewer | | | |
