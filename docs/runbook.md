# Runbook

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `SplitError: patient X appears in both train and test` | manifest edited by hand or regenerated with a different pattern | rebuild with `scripts/build_manifest.py`, do not hand-edit |
| `InputValidationError: image and label do not occupy the same physical space` | mask resaved without the source affine | re-export the mask with the CT affine (TC 02) |
| CUDA out of memory | patch size or batch size too large | drop `data.patch_size` to 64^3, `training.batch_size` to 1, or reduce `model.channels`; rebuild the cache after a patch-size change |
| Validation Dice stuck near 0 | learning rate, label polarity, or empty foreground sampling | check `pos_ratio`, confirm labels are non-empty, try lr 1e-3 on a small subset |
| Slow first epoch | cache being built | expected once per config hash; `data/cache/<hash>/` |
| Mask misaligned in the viewer | wrong export path, or preprocessing changed without re-running inference | inference always exports on the source grid; re-run `src.infer`, check the audit JSON `input_shape` |
| API returns 503 | `LIVER3D_CHECKPOINT` unset or missing | set it to an existing `best.pt` and restart |
| API job stays `queued` | worker died or model failed to load | check server logs, `/healthz` `model_loaded`, restart the service |

## Tuning order (validation only)

1. Spacing from the dataset median, then patch size to the largest that fits memory.
2. HU window: try [-200, 250] and [-100, 200]; keep whichever wins on validation.
3. `pos_ratio` 0.5-0.8; then learning rate 1e-4 vs 3e-4.
4. Threshold and postprocessing (`largest_component`, `min_component_ml`) last, on validation.
5. Freeze everything, then open the test set once.

Changing anything under `data:` changes the cache hash; the cache rebuilds
automatically.

## Performance measurement

Report hardware, volume size, warm-up handling and sliding-window settings with
any timing. `inference_seconds` in `eval_test/case_metrics.csv` measures the
sliding-window pass only; `runtime_seconds` in the inference audit covers
preprocess -> inference -> postprocess -> inverse transform.

## Cache and disk

- Preprocessed cache: `data/cache/<config-hash>/<case>.npz`. Safe to delete.
- API jobs: `LIVER3D_WORK_DIR` (default `outputs/api/jobs/<job_id>/`), plus
  `jobs.db`. Apply the retention policy here.
