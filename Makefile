UV ?= uv
CONFIG ?= configs/models/esfpnet.yaml

.PHONY: setup train predict table2 evaluate baseline

setup:
	@$(UV) run python scripts/download_dataset.py
	@$(UV) run python scripts/download_mit_weights.py

train:
	@$(UV) run python scripts/train.py --config $(CONFIG)

predict:
	@$(UV) run python -m scripts.export_predictions --config $(CONFIG)

table2:
	@$(UV) run python -m scripts.evaluate_predictions

evaluate: predict table2

baseline: train evaluate
