# ADR-002: RAG, Agentic Guardrails, Dual-Backend Evaluation & Kaggle QLoRA Architecture

## Context
Per `NLP_IR.pdf` (*TBP: Frontier LLM Research & Engineering*, Use Case A), the project requires:
- Experimental comparison **B0–B6** using an **identical base model** across **B0–B5**, and comparing **B6** directly against its identical parent (**B4** or **B5**).
- Seven single-variable ablations (**A1–A7**), $\ge 3$ independent test runs, $\ge 20$ red-team cases, blind human adjudication workflow, and live serving UI + API.
- Compute constraints: Local development hardware has a 2 GB VRAM GeForce 920MX (Maxwell) + 11 GB RAM, while QLoRA fine-tuning and open-weight LLM inference (`0.5B–3B`) run on **Kaggle GPU** (T4/P100 16 GB VRAM).

## Decisions
1. **Identical Base Generator Contract (`Qwen/Qwen2.5-1.5B-Instruct` + 4-bit NF4 QLoRA)**:
   - Selected over 7B+ models to fit comfortably within Kaggle's 16 GB VRAM alongside `BAAI/bge-m3` / `BM25` retrieval and `max_seq_length=2048`.
   - `HuggingFaceCausalBackend` in `src/llm_backend.py` enforces identical base weights across B0–B5, attaching `PeftModel` only for B5 (and B6 if selected on `dev`).
2. **Pluggable Local CPU Backend (`ExtractivePolicyBackend` & `TF-IDF + SVD` Dense Index)**:
   - Allows full offline/CPU execution and verification of chunking (A1), BM25/Dense/Hybrid RRF/Reranker retrieval (A2), bounded retry, claim verification, tool guardrails, red-team defenses, and FastAPI UI/API without external proprietary API leakage or 2 GB GPU OOM errors.
   - Kaggle GPU mode is activated simply via `--backend hf` and `requirements-kaggle.txt`.
3. **Pure-Python + Scikit-Learn + `rank-bm25` Indexing**:
   - Chosen over external vector database servers (Qdrant/Milvus) to keep artifacts portable, deterministic, versioned via manifests, and lightweight on Kaggle and local laptops.
4. **Strict Pydantic `ToolGuard` for B6**:
   - All agentic tool calls (`search_corpus`, `get_regulation_metadata`, `check_temporal_validity`, `get_cross_references`) undergo allowlist checking, Pydantic schema validation, path/SQL/shell character rejection, `top_k <= 10`, and `max_calls = 4` budget enforcement.

## Consequences
- Every module is testable offline via `pytest` and serves a working FastAPI UI/API locally with automatic fallback, while `scripts/train_qlora_kaggle.py` and `scripts/run_experiments.py --backend hf` scale directly to Kaggle GPU without changing schemas or data splits.
