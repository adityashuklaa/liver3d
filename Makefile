PY ?= python
CONFIG ?= configs/baseline.yaml
CHECKPOINT ?= outputs/demo/best.pt

.PHONY: install test demo data manifest train eval serve clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

demo:
	$(PY) scripts/quickstart.py

data:
	$(PY) scripts/make_synthetic_dataset.py --out data/synthetic --cases 8

manifest:
	$(PY) scripts/build_manifest.py --root data/synthetic --out data_manifests/synthetic_split.csv

train:
	$(PY) -m src.train --config $(CONFIG)

eval:
	$(PY) -m src.evaluate --checkpoint $(CHECKPOINT) --split test

serve:
	LIVER3D_CHECKPOINT=$(CHECKPOINT) uvicorn src.api.service:app --port 8000

clean:
	rm -rf data/cache outputs/api .pytest_cache
