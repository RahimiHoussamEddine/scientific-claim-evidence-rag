# Scientific Claim Evidence Assistant

A local retrieval-augmented system that searches scientific abstracts for evidence relevant to a claim and asks a small open-weight model for a verdict. The interface places the retrieved abstracts beside the model output so users can inspect the source evidence.

This is an evaluation and research-assistance prototype, not a systematic review, medical advice, or an authoritative scientific fact checker. Hybrid retrieval performed well on the labeled development split, but FLAN-T5-small produced mostly unparseable verdicts. Do not rely on its generated label; the working product value is evidence retrieval and human inspection. SciFact abstracts are a small benchmark and cannot represent all available studies or resolve scientific consensus.

## Why this project

P1–P15 already cover semantic job retrieval, recommendation ranking, fairness, causal uplift, image classification, and statistical forecasting. This project adds a local RAG workflow with sparse/dense/hybrid retrieval comparisons, a pinned open model, visible source abstracts and IDs, a Streamlit interface, and API endpoints.

## Dataset and licenses

The pipeline downloads the original [SciFact release](https://huggingface.co/datasets/allenai/scifact) archive from AllenAI's S3 endpoint. It contains 5,183 abstracts, 809 labeled training claims, 300 labeled development claims, and 300 unlabeled test claims. Raw and extracted files stay under data/ and are ignored by Git. The dataset is licensed CC BY-NC 2.0; users must review and follow its non-commercial terms. The project does not redistribute the dataset. The benchmark was introduced in the [SciFact paper](https://aclanthology.org/2020.emnlp-main.609/).

The default generator is [google/flan-t5-small](https://huggingface.co/google/flan-t5-small), loaded locally at a pinned Hugging Face revision under Apache-2.0. The sentence encoder is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), also Apache-2.0. Weights are downloaded at setup/runtime and are not committed. Project code is MIT licensed. See data/source_manifest.json and NOTICE.md for provenance and separate license terms.

## Setup

Run these commands from this repository directory. The virtual environment, package cache, dataset, model cache, and generated reports remain inside the project directory.

~~~bash
python3.12 -m venv .venv
PIP_CACHE_DIR=.cache/pip .venv/bin/python -m pip install --upgrade pip
make install-dev PYTHON=.venv/bin/python
make install-llm PYTHON=.venv/bin/python
make fetch PYTHON=.venv/bin/python
make build PYTHON=.venv/bin/python
~~~

The dense encoder is fetched during `make build`; FLAN-T5-small is fetched during the first full `make evaluate`. Internet is needed for those first downloads. The app and API targets then run with Hugging Face offline mode and use the local caches. Run the full evaluation once before the first model-backed app/API interaction.

## Evaluate

~~~bash
make evaluate PYTHON=.venv/bin/python
make test PYTHON=.venv/bin/python
make quality PYTHON=.venv/bin/python
~~~

Evaluation reports retrieval Recall@5, Precision@5, MRR, and nDCG@5 for BM25, dense retrieval, and reciprocal-rank fusion on the official labeled development split. It also evaluates FLAN-T5-small over all 300 dev claims. After a retrieval report exists, `make evaluate-generation PYTHON=.venv/bin/python` reruns only generation and reuses the cached dense index. The hidden test split is unlabeled and is never scored. The dev results are validation results, not an untouched test estimate.

### Observed development results

Retrieval metrics below are averaged over the 188 dev claims with annotated evidence; each method returns five abstracts. The other 112 claims have no annotated evidence.

| Retriever | Evidence Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.879 | 0.186 | 0.777 | 0.798 |
| Dense | 0.869 | 0.187 | 0.721 | 0.756 |
| Hybrid RRF | **0.922** | **0.196** | **0.795** | **0.823** |

FLAN-T5-small reached 0.060 accuracy and 0.085 macro-F1 across the three labels present in the dev set; 86.3% of generated outputs could not be parsed as an explicit verdict. It emitted 95 bracketed IDs: 94.7% referred to a retrieved abstract, while 6.3% matched an annotated gold evidence document. These citation checks do not establish factual correctness. The model is an evaluated but unsuccessful verdict component; inspect the retrieved abstracts directly. Detailed aggregates are in `reports/validation_metrics.json`.

## Run the interface

~~~bash
make app PYTHON=.venv/bin/python
~~~

Open http://localhost:8501. Choose an example or enter a claim, select BM25/dense/hybrid retrieval, and run the local model. The model receives up to the first three retrieved abstracts; the evidence cards display all retrieved abstracts and IDs. A warning displays the measured dev-set generation limitations.

## API

~~~bash
make api PYTHON=.venv/bin/python
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/search -H 'Content-Type: application/json' -d '{"claim":"A specific scientific claim with enough detail.","method":"hybrid","k":5}'
curl -X POST http://localhost:8000/v1/verify -H 'Content-Type: application/json' -d '{"claim":"A specific scientific claim with enough detail.","method":"hybrid","k":5}'
~~~

Interactive API docs are at http://localhost:8000/docs. API requests are served on loopback only by default.

## Reproducibility and performance

The data archive checksum, model identifier/revision, record counts, and license are in data/source_manifest.json. All downloads and model caches are project-local. CPU inference is single-process with a bounded thread count. No model fine-tuning is performed. The dense index is cached locally and ignored by Git.

Docker is intentionally omitted: packaging the CPU PyTorch stack and model weights would make the image unnecessarily large. GitHub Actions configuration runs static checks and focused tests without downloading benchmark data or model weights.

## Limitations

- SciFact covers a small, expert-annotated set of claims and abstracts; it is not a full-literature search.
- The dataset is CC BY-NC 2.0, so commercial use requires separate rights. The repository contains no dataset files.
- FLAN-T5-small can misunderstand evidence, emit an invalid verdict, or invent a citation. Retrieved abstracts and model output must be checked by a human.
- Citation-source validity checks whether a referenced ID was retrieved; citation evidence precision checks whether the ID matches an annotated gold evidence document. Neither is a general factuality score.
- The labeled dev split is used for evaluation. No claim is made about performance on the hidden unlabeled test split or current scientific literature.
