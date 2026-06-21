UV ?= uv
CONFIG ?= configs/models/esfpnet.yaml
EXP ?= $(exp)
RUN_CONFIG = $(if $(EXP),configs/experiments/$(EXP).yaml,$(CONFIG))
MS_EXP ?= $(if $(EXP),$(EXP),MS_baseline_rgb)
MS_CONFIG ?= configs/experiments/$(MS_EXP).yaml
MS_CHECKPOINT ?= results/$(MS_EXP)/best.pth
MS_DEVICE ?= auto
CROSS_SUBSETS ?= cvc-clinicdb cvc-colondb etis-larib
MULTISOURCE_EXPERIMENTS := MS_baseline_rgb C2_consistency_multisource FD2_dino_distill_multisource
RECALL_CONFIG ?= $(MS_CONFIG)
RECALL_CHECKPOINT ?= $(MS_CHECKPOINT)
RECALL_DEVICE ?= $(MS_DEVICE)
RECALL_BATCH_SIZE ?= 8
RECALL_NUM_WORKERS ?= 4
RECALL_ECE_BINS ?= 15
RECALL_THRESHOLDS ?= 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60
RECALL_OUTPUT_ROOT ?= outputs/recall_safe_calibration
RECALL_TEST_SOURCE ?= etis_larib
RECALL_VAL_ARGS = $(if $(RECALL_VAL_IMAGES),--val-images $(RECALL_VAL_IMAGES) --val-masks $(RECALL_VAL_MASKS),)
RECALL_TEST_ARGS = $(if $(RECALL_TEST_IMAGES),--test-images $(RECALL_TEST_IMAGES) --test-masks $(RECALL_TEST_MASKS),)


.DEFAULT_GOAL := $(if $(EXP),exp,setup)

.PHONY: setup train predict table2 evaluate baseline exp cross train-ms eval-ms smoke-ms-train smoke-ms-eval smoke-ms ms recall-safe-checkpoint recall-safe-calibration

setup:
	@$(UV) run python scripts/download_dataset.py
	@$(UV) run python scripts/download_mit_weights.py
	@$(UV) run python scripts/generate_pc_endonorm_dataset.py \
		--dataset-name Kvasir-SEG \
		--images-dir data/raw/kvasir-seg/images \
		--output-dir data/preprocessed/kvasir-seg \
		--image-size 352 \
		--config configs/preprocess_pc_endonorm.yaml
	@$(UV) run python scripts/prepare_cross_datasets.py --subsets $(CROSS_SUBSETS)

train:
	@$(UV) run python scripts/train.py --config $(RUN_CONFIG)

predict:
	@$(UV) run python -m scripts.export_predictions --config $(RUN_CONFIG)

table2:
	@$(UV) run python -m scripts.evaluate_predictions

evaluate: predict table2

baseline: train evaluate

ifneq ($(filter $(EXP),$(MULTISOURCE_EXPERIMENTS)),)
exp: ms
else
exp: train evaluate cross
endif

cross:
	@$(UV) run python scripts/eval_cross_dataset.py $(if $(EXP),--experiments $(EXP),)

classical:
	@$(UV) run python -m scripts.classical_segmentation
	@$(UV) run python -m scripts.classical_segmentation --external --data-root data/raw/cvc-colondb --output-root outputs/predictions_cross/cvc-colondb
	@$(UV) run python -m scripts.classical_segmentation --external --data-root data/raw/etis-larib --output-root outputs/predictions_cross/etis-larib
	@$(UV) run python -m scripts.evaluate_predictions
	@$(UV) run python -m scripts.evaluate_predictions --external --data-root data/raw/cvc-colondb --predictions-root outputs/predictions_cross/cvc-colondb --output-name table2_cvc-colondb
	@$(UV) run python -m scripts.evaluate_predictions --external --data-root data/raw/etis-larib --predictions-root outputs/predictions_cross/etis-larib --output-name table2_etis-larib

train-ms:
	@$(UV) run python scripts/train.py --config $(MS_CONFIG)

eval-ms:
	@$(UV) run python scripts/eval_multisource_crossdataset.py \
		--config $(MS_CONFIG) \
		--checkpoint $(MS_CHECKPOINT) \
		--device $(MS_DEVICE)

smoke-ms-train:
	@$(UV) run python scripts/train.py --config $(MS_CONFIG) --smoke_test

smoke-ms-eval:
	@$(UV) run python scripts/eval_multisource_crossdataset.py \
		--config $(MS_CONFIG) \
		--checkpoint $(MS_CHECKPOINT) \
		--device $(MS_DEVICE) \
		--smoke_test

smoke-ms: smoke-ms-train smoke-ms-eval

ms: train-ms eval-ms

recall-safe-checkpoint:
	@if [ ! -f "$(RECALL_CHECKPOINT)" ]; then \
		echo "Checkpoint not found: $(RECALL_CHECKPOINT)"; \
		echo "Training with config: $(RECALL_CONFIG)"; \
		$(UV) run python scripts/train.py --config $(RECALL_CONFIG); \
	fi; \
	if [ ! -f "$(RECALL_CHECKPOINT)" ]; then \
		echo "Expected checkpoint still missing after training: $(RECALL_CHECKPOINT)"; \
		exit 1; \
	fi; \
	echo "Using checkpoint: $(RECALL_CHECKPOINT)"

recall-safe-calibration: recall-safe-checkpoint
	@$(UV) run python experiments/recall_safe_calibration/run_pipeline.py \
		--config $(RECALL_CONFIG) \
		--checkpoint $(RECALL_CHECKPOINT) \
		--device $(RECALL_DEVICE) \
		--batch-size $(RECALL_BATCH_SIZE) \
		--num-workers $(RECALL_NUM_WORKERS) \
		--ece-bins $(RECALL_ECE_BINS) \
		--thresholds $(RECALL_THRESHOLDS) \
		--test-source-name $(RECALL_TEST_SOURCE) $(RECALL_VAL_ARGS) $(RECALL_TEST_ARGS) \
		--output-root $(RECALL_OUTPUT_ROOT)
