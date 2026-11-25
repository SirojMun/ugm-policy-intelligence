# UGM Academic Policy Intelligence — Research-Grade RAG & Agentic LLM System

End-to-end research & engineering pipeline for **TBP: Frontier LLM Research & Engineering (Use Case A: Academic Policy Intelligence)** built on **422 public Universitas Gadjah Mada (UGM) regulations & academic/service guidelines**.

## 1. Corpus & Scraping Pipeline (Stage 1–3)
- `sources_config.yaml`, `01_crawl_metadata.py`, `02_download_and_hash.py`, `03_extract_and_filter.py`, `ugm_policy_pipeline/`
- Produces `data/raw_pdfs/` (422 clean PDFs), `corpus_md/` (422 hierarchical legal Markdown files), `corpus_clean.jsonl`, `source_register.csv/.jsonl`, `skipped_scanned_pdfs.csv`, `data/logs/deduplicated.csv`, and `data/logs/flagged_pii_docs.csv`.

## 2. Research Architecture (`src/`, `prompts/`, `red_team/`, `scripts/`, `app/`)
- **Chunking & Retrieval (`src/chunking.py`, `src/retrieval.py`)**:
  - Ablation **A1**: `fixed` vs `heading_aware` vs `hierarchical` (parent `BAB` + child `Pasal`).
  - Ablation **A2**: `dense` vs `bm25` vs `hybrid` (RRF) vs `hybrid_rerank` + metadata filters (`Recall@5/10`, `nDCG@10`, `MRR@10`).
- **Benchmark, Qrels & Blind Human Review (`src/benchmark.py`, `src/human_eval.py`, `data/benchmark/`)**:
  - `train.jsonl` (60), `dev.jsonl` (30), `test_locked.jsonl` (110 $\ge 100$), `qrels.jsonl/.tsv` (239), `split_manifest.json` (SHA-256 + `leakage_free: true`).
  - Blind annotation sheets `data/annotation/annotator_1_blind.jsonl` & `annotator_2_blind.jsonl` + Cohen's $\kappa$ & disagreement adjudication workflow.
- **Configurations B0–B6 & Ablations A1–A7 (`prompts/registry.yaml`, `src/agents_pipeline.py`, `src/eval_harness.py`)**:
  - Same base model across **B0–B5**; **B6** Guarded Agentic LLM with `ToolGuard` schema validation, temporal verification, and claim verifier.
  - **Red-Team Suite (`red_team/red_team_suite.jsonl`, `src/red_team_runner.py`)**: 20 categorized adversarial fixtures.
- **Kaggle GPU QLoRA Fine-Tuning (`scripts/train_qlora_kaggle.py`, `requirements-kaggle.txt`)**:
  - Prepares leak-free `data/sft/train_sft.jsonl` from `train.jsonl` and trains 4-bit QLoRA on `Qwen/Qwen2.5-1.5B-Instruct` (`r=16`, `alpha=32`, batch `2x8`, `max_seq_length=2048`).
- **Serving UI & API (`app/server.py`, `app/static/index.html`)**:
  - FastAPI UI + `/api/query` + `/health` + `/api/traces` with JSONL observability logging and BM25 fallback.

## 3. Running Verification, Experiments & Serving

```bash
# Run full unit & integration test suite
.venv/bin/pytest -v

# Prepare leak-free SFT training data locally (CPU)
.venv/bin/python scripts/train_qlora_kaggle.py --prepare-only

# Package lightweight zip bundle (~6-10 MB) for Kaggle Dataset upload
.venv/bin/python scripts/package_for_kaggle.py

# Run A1-A7 ablations, B0-B6 (3 independent runs), and Red-Team suite locally (extractive baseline)
.venv/bin/python scripts/run_experiments.py --backend extractive --out-dir reports

# On Kaggle GPU (recommended: import notebooks/kaggle_ugm_policy_pipeline.ipynb + upload dist/ugm_policy_kaggle_bundle.zip):
python scripts/train_qlora_kaggle.py --base-model Qwen/Qwen2.5-1.5B-Instruct --output-dir models/b5_qlora_adapter
python scripts/run_experiments.py --backend hf --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter-dir models/b5_qlora_adapter --dense-model intfloat/multilingual-e5-small --reranker-model BAAI/bge-reranker-base --out-dir reports_kaggle

# Start live UI & API server
.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 8000
```
