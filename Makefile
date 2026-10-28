PY ?= python3
CFG ?= configs/cpu_smoke.yaml
# Set an HF mirror if huggingface.co is unreachable, e.g. make all EP=https://hf-mirror.com
EP ?=
FLAGS = $(if $(EP),--hf-endpoint $(EP),)

.PHONY: help prepare index baselines train eval all clean

help:
	@echo "targets: prepare | index | baselines | train | eval | all | clean"
	@echo "example: make all CFG=configs/default.yaml EP=https://hf-mirror.com"

prepare:
	$(PY) scripts/prepare_data.py --config $(CFG) $(FLAGS)

index:
	$(PY) scripts/build_index.py --config $(CFG) $(FLAGS)

## baselines: prompt-only and RAG, no training required
baselines:
	$(PY) scripts/evaluate.py --config $(CFG) $(FLAGS) --arms prompt_only rag

train:
	$(PY) scripts/train_sft.py --config $(CFG) $(FLAGS)

## eval: full four-way comparison plus the oracle arm
eval:
	$(PY) scripts/evaluate.py --config $(CFG) $(FLAGS) --arms prompt_only rag sft sft_rag rag_gold

all: prepare index baselines train eval

clean:
	rm -rf data checkpoints results
