# Checklists

## Before training

- [ ] Data authorization and license recorded in `docs/data_card.md`
- [ ] Direct identifiers removed from files, headers and filenames
- [ ] Patient-level split locked (`scripts/build_manifest.py`) and checksummed
- [ ] `validate_manifest` passes (no duplicates, no cross-split patients, files exist)
- [ ] Image/label geometry validated (automatic when the cache is built)
- [ ] Label definition documented (which raw values map to liver)
- [ ] Baseline configuration committed (`configs/baseline.yaml`)
- [ ] GPU memory dry run: one training step at the chosen patch size and batch size
- [ ] Output location, cache location and retention defined

Dry run command:

```bash
python -m src.train --config configs/baseline.yaml --max-epochs 1 --iters-per-epoch 5 --limit-cases 2
```

## Before test evaluation

- [ ] Model selection rule frozen (`training.checkpoint_metric`, best validation Dice)
- [ ] Test set not used for tuning, threshold selection or early stopping
- [ ] Checkpoint SHA-256 recorded (`run_summary.json`)
- [ ] Evaluation script exercised on the validation split first
- [ ] Metric definitions and units fixed (Dice, IoU, precision, recall, HD95 mm, ASSD mm)
- [ ] Exclusion rules fixed and written down before results are seen
- [ ] Output folder empty or versioned

## Before demonstration or deployment

- [ ] Source geometry restoration verified (mask opens aligned in an independent viewer)
- [ ] Worst-case examples reviewed with a clinician
- [ ] Unsupported and out-of-range inputs rejected or flagged (test with a wrong-spacing volume)
- [ ] Viewer or client shows model version, warnings and the review action
- [ ] Accept / correct / reject workflow tested end to end (`POST .../review`)
- [ ] Access controls tested, including a denied unauthorized request
- [ ] Logs and audit rows contain no identifiers
- [ ] Rollback bundle available (previous checkpoint + config + validation report)
- [ ] Academic prototype limitation displayed to every user

## Per release

- [ ] `docs/model_card.md` has no remaining `TO_CONFIRM`
- [ ] `docs/validation_report_template.md` filled and signed off
- [ ] `pytest -q` green on the release commit
- [ ] Environment record archived (`env.json`)
- [ ] Release bundle stored with its checksum
