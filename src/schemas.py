"""Strict Pydantic schemas for corpus chunks, benchmark/qrels, B0-B6 outputs, and run manifests."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """Single indexed unit produced by fixed, heading-aware, or hierarchical chunkers."""

    chunk_id: str
    doc_id: str
    parent_id: str = ""
    strategy: Literal["fixed", "heading_aware", "hierarchical"]
    level: Literal["single", "parent", "child"] = "single"
    title: str
    doc_type: str
    year: int | None = None
    valid_time: str = ""
    status: str = "Unknown"
    version: str = "original"
    usage_rights: str = "Public Academic Regulation - UGM"
    section_path: str = ""
    pasal: str = ""
    word_count: int
    char_count: int
    text: str


class Citation(BaseModel):
    """Grounded passage/pasal citation required by the Academic Policy Intelligence contract."""

    doc_id: str
    chunk_id: str
    pasal: str = ""
    section_path: str = ""
    valid_time: str = ""
    status: str = "Unknown"
    quote: str = ""


class AtomicClaim(BaseModel):
    """Decomposed factual/normative claim verified against retrieved passages."""

    claim_id: str
    statement: str
    citations: list[str] = Field(default_factory=list)
    verdict: Literal["supported", "unknown"] = "unknown"


class TraceSpan(BaseModel):
    """Execution observability span for retrieval, verification, and tool calls."""

    step: str
    detail: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyResponse(BaseModel):
    """Minimum required output contract across all B0-B6 configurations."""

    query_id: str = ""
    config_id: str
    query: str
    answer: str
    atomic_claims: list[AtomicClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    valid_time: str = ""
    verdict: Literal["supported", "unknown"] = "unknown"
    abstain_reason: str = ""
    latency_ms: float = 0.0
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0
    model_id: str = ""
    prompt_version: str = ""
    index_version: str = ""
    fallback_used: bool = False
    trace: list[TraceSpan] = Field(default_factory=list)


class BenchmarkQuery(BaseModel):
    """Single query instance in train, dev, or locked test splits."""

    query_id: str
    split: Literal["train", "dev", "test"]
    slice_type: Literal["easy", "multi-hop", "conflict", "no-evidence"]
    answerability: Literal["answerable", "unanswerable"]
    query: str
    as_of_date: str = "2026-11-25"
    gold_answer: str = ""
    gold_valid_time: str = ""
    gold_verdict: Literal["supported", "unknown"] = "supported"
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_pasals: list[str] = Field(default_factory=list)
    conflict_pair: list[str] = Field(default_factory=list)
    annotation_rule: str = ""


class QrelEntry(BaseModel):
    """Passage/document relevance judgment for retrieval metrics (Recall@k, nDCG@10, MRR@10)."""

    query_id: str
    doc_id: str
    chunk_id: str
    pasal: str = ""
    relevance: int = Field(ge=0, le=2)
    doc_version: str = "original"
    annotation_rule: str = "exact_source_passage_match"
