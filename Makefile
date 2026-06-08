UV ?= uv
CONFIG ?= configs/models/esfpnet.yaml

.PHONY: setup train table2

setup:
	@$(UV) run python scripts/download_dataset.py
	@$(UV) run python scripts/download_mit_weights.py

train:
	@$(UV) run python scripts/train.py --config $(CONFIG)

table2:
	@$(UV) run python -m scripts.evaluate_predictions
