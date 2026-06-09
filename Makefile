UV ?= uv
CONFIG ?= configs/models/esfpnet.yaml
EXP ?=
RUN_CONFIG = $(if $(EXP),configs/experiments/$(EXP).yaml,$(CONFIG))

.PHONY: setup train predict table2 evaluate baseline exp

setup:
	@$(UV) run python scripts/download_dataset.py
	@$(UV) run python scripts/download_mit_weights.py

train:
	@$(UV) run python scripts/train.py --config $(RUN_CONFIG)

predict:
	@$(UV) run python -m scripts.export_predictions --config $(RUN_CONFIG)

table2:
	@$(UV) run python -m scripts.evaluate_predictions

evaluate: predict table2

baseline: train evaluate

exp: train evaluate
