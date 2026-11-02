PYTHON ?= .venv/bin/python
PIP_CACHE_DIR := $(CURDIR)/.cache/pip
export HF_HOME := $(CURDIR)/.cache/huggingface
export HF_HUB_CACHE := $(CURDIR)/.cache/huggingface/hub
export TRANSFORMERS_CACHE := $(CURDIR)/.cache/huggingface/transformers
export SENTENCE_TRANSFORMERS_HOME := $(CURDIR)/.cache/sentence-transformers
export TORCH_HOME := $(CURDIR)/.cache/torch
export TOKENIZERS_PARALLELISM := false

.PHONY: install-dev install-llm fetch build evaluate-retrieval evaluate-generation evaluate test quality app api clean

install-dev:
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -e '.[app,dev]'

install-llm:
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -e '.[llm]'

fetch:
	$(PYTHON) -m scientific_evidence_rag.fetch

build:
	$(PYTHON) -m scientific_evidence_rag.build_index

evaluate:
	$(PYTHON) -m scientific_evidence_rag.evaluate --generation

evaluate-retrieval:
	$(PYTHON) -m scientific_evidence_rag.evaluate

evaluate-generation:
	HF_HUB_OFFLINE=1 $(PYTHON) -m scientific_evidence_rag.evaluate --generation-only

test:
	$(PYTHON) -m pytest

quality:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

app:
	HF_HUB_OFFLINE=1 $(PYTHON) -m streamlit run app.py --server.headless=true

api:
	HF_HUB_OFFLINE=1 $(PYTHON) -m uvicorn scientific_evidence_rag.api:app --app-dir src --host 127.0.0.1 --port 8000

clean:
	rm -rf .cache data/processed reports/validation_metrics.json
