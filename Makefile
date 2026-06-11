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

.DEFAULT_GOAL := $(if $(EXP),exp,setup)

.PHONY: setup train predict table2 evaluate baseline exp cross train-ms eval-ms smoke-ms-train smoke-ms-eval smoke-ms ms

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
	@$(UV) run python -m scripts.debug_robust_augmentations
	@$(UV) run python -m scripts.debug_robust_augmentations --config configs/experiments/K2_robust_full_randconv.yaml

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
