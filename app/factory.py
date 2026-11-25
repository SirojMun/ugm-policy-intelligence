"""FastAPI application factory with UI, structured JSONL tracing, and BM25 fallback."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agents_pipeline import PolicyExperimentRunner
from src.chunking import chunk_corpus, load_corpus
from src.eval_harness import load_jsonl_rows
from src.llm_backend import BaseLLMBackend, ExtractivePolicyBackend, HuggingFaceCausalBackend
from src.retrieval import PolicyRetrievalIndex
from src.schemas import PolicyResponse, TraceSpan

TRACE_LOG_PATH = Path("logs/api_traces.jsonl")
STATIC_INDEX_PATH = Path(__file__).parent / "static" / "index.html"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    config_id: Literal["B0", "B1", "B2", "B3", "B4", "B5", "B6"] = "B6"
    as_of_date: str = Field(default="2026-11-25", pattern=r"^\d{4}-\d{2}-\d{2}$")
    top_k: int = Field(default=5, ge=1, le=10)
    max_context_tokens: int = Field(default=4000, ge=500, le=8000)


def create_app(
    corpus_jsonl: str | Path = "corpus_clean.jsonl",
    train_jsonl: str | Path = "data/benchmark/train.jsonl",
    backend_mode: str | None = None,
    trace_log_path: str | Path = TRACE_LOG_PATH,
) -> FastAPI:
    app = FastAPI(title="UGM Academic Policy Intelligence API", version="1.0.0")
    trace_file = Path(trace_log_path)

    docs = load_corpus(corpus_jsonl)
    docs_by_id = {d["doc_id"]: d for d in docs}
    idx_fixed = PolicyRetrievalIndex(chunk_corpus(docs, "fixed"))
    idx_head = PolicyRetrievalIndex(chunk_corpus(docs, "heading_aware"))
    idx_hier = PolicyRetrievalIndex(chunk_corpus(docs, "hierarchical"))

    train_examples: list[dict[str, str]] = []
    if Path(train_jsonl).exists():
        train_rows = load_jsonl_rows(train_jsonl)
        train_examples = [
            {"query": r["query"], "answer": r["gold_answer"]}
            for r in train_rows
            if r.get("answerability") == "answerable"
        ][:3]

    mode = backend_mode or os.getenv("UGM_POLICY_BACKEND", "extractive")
    base_model = os.getenv("UGM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_dir = os.getenv("UGM_QLORA_ADAPTER")

    base_be: BaseLLMBackend
    peft_be: BaseLLMBackend
    if mode == "hf":
        try:
            base_be = HuggingFaceCausalBackend(base_model, adapter_dir=None)
            peft_be = HuggingFaceCausalBackend(base_model, adapter_dir=adapter_dir) if adapter_dir else base_be
        except Exception:
            base_be = ExtractivePolicyBackend()
            peft_be = ExtractivePolicyBackend(adapter_tag="qlora-fallback")
    else:
        base_be = ExtractivePolicyBackend()
        peft_be = ExtractivePolicyBackend(adapter_tag="qlora-train-v1")

    runner = PolicyExperimentRunner(
        docs_by_id=docs_by_id,
        fixed_index=idx_fixed,
        heading_index=idx_head,
        hierarchical_index=idx_hier,
        base_backend=base_be,
        peft_backend=peft_be,
        train_examples=train_examples,
    )

    def _log_trace(req: QueryRequest, resp: PolicyResponse) -> None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_id": req.config_id,
            "query": req.query,
            "as_of_date": req.as_of_date,
            "verdict": resp.verdict,
            "valid_time": resp.valid_time,
            "abstain_reason": resp.abstain_reason,
            "citations_count": len(resp.citations),
            "latency_ms": resp.latency_ms,
            "fallback_used": resp.fallback_used,
            "trace": [t.model_dump() for t in resp.trace],
        }
        with trace_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @app.get("/", response_class=HTMLResponse)
    def serve_ui() -> HTMLResponse:
        if STATIC_INDEX_PATH.exists():
            return HTMLResponse(STATIC_INDEX_PATH.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>UGM Academic Policy Intelligence API</h1>")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "corpus_documents": len(docs),
            "hierarchical_searchable_chunks": len(idx_hier.chunks),
            "dense_backend": idx_hier.dense_backend,
            "base_model": base_be.model_id,
            "peft_model": peft_be.model_id,
        }

    @app.post("/api/query", response_model=PolicyResponse)
    def query_policy(req: QueryRequest) -> PolicyResponse:
        try:
            resp = runner.run(
                config_id=req.config_id,
                query=req.query,
                as_of_date=req.as_of_date,
                top_k=req.top_k,
                max_context_tokens=req.max_context_tokens,
            )
        except Exception as exc:
            hits = idx_hier.retrieve(req.query, mode="bm25", top_k=3)
            resp = PolicyResponse(
                config_id=req.config_id,
                query=req.query,
                answer="Terjadi gangguan pada backend generasi; berikut fallback aman atau penolakan terverifikasi."
                if not hits
                else f"Fallback BM25 ({hits[0].chunk.doc_id}): {hits[0].chunk.text[:280]}",
                verdict="unknown",
                abstain_reason=f"fallback_triggered:{type(exc).__name__}",
                fallback_used=True,
                trace=[TraceSpan(step="fallback_handler", detail=str(exc))],
            )
        _log_trace(req, resp)
        return resp

    @app.get("/api/traces")
    def get_recent_traces(limit: int = 20) -> dict[str, Any]:
        if not trace_file.exists():
            return {"traces": []}
        lines = [l for l in trace_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        return {"traces": [json.loads(l) for l in lines[-limit:][::-1]]}

    return app
