"""Ablation A2 & B2-B6 Retrieval Engine: BM25, Dense, Hybrid RRF, Reranking, and Evaluation Metrics."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.schemas import ChunkRecord, QrelEntry

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SHARED_ENCODER_CACHE: dict[tuple[str, str], Any] = {}


def tokenize_id(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer suitable for Indonesian legal terminology."""
    return TOKEN_RE.findall((text or "").lower())


@dataclass
class RetrievalConfig:
    """Explicit retrieval backend and strict-mode experiment configuration."""

    embedding_backend: str = "auto"
    embedding_model: str | None = None
    reranker_backend: str = "auto"
    reranker_model: str | None = None
    fallback_enabled: bool = True
    strict_experiment_mode: bool = False


@dataclass
class RetrievalFilters:
    """Optional metadata filters for B3-B6 temporal/category-aware retrieval."""

    doc_type: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    exclude_statuses: tuple[str, ...] = ()
    only_statuses: tuple[str, ...] = ()


@dataclass
class ScoredChunk:
    """Retrieved chunk paired with score and retriever provenance."""

    chunk: ChunkRecord
    score: float
    source_retriever: str
    parent_text: str = ""


def matches_year(chunk: ChunkRecord, year_min: int | None = None, year_max: int | None = None) -> bool:
    """Enforce strict year bounds so unknown-year chunks never bypass temporal filters."""
    if year_min is None and year_max is None:
        return True
    if chunk.year is None:
        return False
    if year_min is not None and chunk.year < year_min:
        return False
    if year_max is not None and chunk.year > year_max:
        return False
    return True


def _matches_filters(chunk: ChunkRecord, filters: RetrievalFilters | None) -> bool:
    if filters is None:
        return True
    if filters.doc_type and chunk.doc_type.lower() != filters.doc_type.lower():
        return False
    if not matches_year(chunk, filters.year_min, filters.year_max):
        return False
    if filters.exclude_statuses and chunk.status in filters.exclude_statuses:
        return False
    if filters.only_statuses and chunk.status not in filters.only_statuses:
        return False
    return True


class PolicyRetrievalIndex:
    """Unified index supporting BM25, Dense (SentenceTransformer or LSA CPU), Hybrid RRF, and Reranking."""

    def __init__(
        self,
        chunks: list[ChunkRecord],
        dense_model_name: str | None = None,
        reranker_model_name: str | None = None,
        lsa_dim: int = 256,
        config: RetrievalConfig | None = None,
        strict_mode: bool = False,
    ) -> None:
        cfg = config or RetrievalConfig(
            embedding_backend="sentence_transformers" if dense_model_name else "tfidf_svd_lsa",
            embedding_model=dense_model_name,
            reranker_backend="cross_encoder" if reranker_model_name else "lexical_cross_scorer",
            reranker_model=reranker_model_name,
            fallback_enabled=not strict_mode,
            strict_experiment_mode=strict_mode,
        )
        eff_dense_model = dense_model_name or cfg.embedding_model
        eff_rerank_model = reranker_model_name or cfg.reranker_model
        self.strict_experiment_mode = bool(strict_mode or cfg.strict_experiment_mode or not cfg.fallback_enabled)
        self.embedding_model = eff_dense_model or "tfidf_svd_lsa"
        self.reranker_model = eff_rerank_model or "lexical_cross_scorer"
        self.configured_dense_backend = "sentence_transformers" if eff_dense_model else "tfidf_svd_lsa"
        self.configured_reranker_backend = "cross_encoder" if eff_rerank_model else "lexical_cross_scorer"
        self.dense_fallback_used = False
        self.reranker_fallback_used = False

        # Index leaf/searchable chunks (child or single), keeping parent lookup table for hierarchical expansion
        self.parent_map: dict[str, ChunkRecord] = {
            c.chunk_id: c for c in chunks if c.level == "parent"
        }
        self.chunks: list[ChunkRecord] = [c for c in chunks if c.level != "parent"]
        if not self.chunks:
            self.chunks = list(chunks)

        # 1. Build BM25 index
        tokenized_docs = [tokenize_id(f"{c.title} {c.section_path} {c.text}") for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_docs)

        # 2. Build Dense index (SentenceTransformer when requested; fail loudly in strict mode)
        self.dense_backend = "tfidf_svd_256"
        self._st_model = None
        self._tfidf = None
        self._svd = None

        if eff_dense_model:
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                cache_key = (f"st:{eff_dense_model}", device)
                if cache_key not in _SHARED_ENCODER_CACHE:
                    m = SentenceTransformer(eff_dense_model, device=device)
                    if device == "cuda":
                        m = m.half()
                    _SHARED_ENCODER_CACHE[cache_key] = m
                self._st_model = _SHARED_ENCODER_CACHE[cache_key]
                texts = [f"{c.section_path}\n{c.text[:1500]}" for c in self.chunks]
                emb = self._st_model.encode(
                    texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
                )
                self.dense_matrix = np.asarray(emb, dtype=np.float32)
                self.dense_backend = f"sentence_transformers:{eff_dense_model}"
            except Exception as exc:
                self._st_model = None
                if self.strict_experiment_mode:
                    raise RuntimeError(
                        f"Strict retrieval mode failed loading dense model '{eff_dense_model}': {exc}"
                    ) from exc
                self.dense_fallback_used = True

        if self._st_model is None:
            texts = [f"{c.title} {c.section_path} {c.text}" for c in self.chunks]
            self._tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
            tfidf_mat = self._tfidf.fit_transform(texts)
            actual_dim = max(2, min(lsa_dim, tfidf_mat.shape[0] - 1, tfidf_mat.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=actual_dim, random_state=42)
            emb = self._svd.fit_transform(tfidf_mat)
            self.dense_matrix = normalize(emb, norm="l2").astype(np.float32)
            self.dense_backend = f"tfidf_svd_{actual_dim}"

        # 3. Optional Cross-Encoder Reranker
        self._cross_encoder = None
        self.reranker_backend = "lexical_cross_scorer"
        if eff_rerank_model:
            try:
                import torch
                from sentence_transformers import CrossEncoder

                device = "cuda" if torch.cuda.is_available() else "cpu"
                cache_key = (f"ce:{eff_rerank_model}", device)
                if cache_key not in _SHARED_ENCODER_CACHE:
                    _SHARED_ENCODER_CACHE[cache_key] = CrossEncoder(eff_rerank_model, device=device, max_length=512)
                self._cross_encoder = _SHARED_ENCODER_CACHE[cache_key]
                self.reranker_backend = f"cross_encoder:{eff_rerank_model}"
            except Exception as exc:
                self._cross_encoder = None
                if self.strict_experiment_mode:
                    raise RuntimeError(
                        f"Strict retrieval mode failed loading reranker '{eff_rerank_model}': {exc}"
                    ) from exc
                self.reranker_fallback_used = True

        self.fallback_used = bool(self.dense_fallback_used or self.reranker_fallback_used)

    def retrieval_metadata(self) -> dict[str, Any]:
        """Expose configured vs actual retrieval backends and fallback provenance."""
        return {
            "configured_backend": self.configured_dense_backend,
            "actual_backend": self.dense_backend,
            "embedding_model": self.embedding_model,
            "reranker_configured_backend": self.configured_reranker_backend,
            "reranker_actual_backend": self.reranker_backend,
            "reranker_model": self.reranker_model,
            "fallback_used": self.fallback_used,
            "strict_experiment_mode": self.strict_experiment_mode,
        }

    def _encode_query(self, query: str) -> np.ndarray:
        if self._st_model is not None:
            vec = self._st_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            return np.asarray(vec, dtype=np.float32)
        assert self._tfidf is not None and self._svd is not None
        q_tfidf = self._tfidf.transform([query])
        q_emb = self._svd.transform(q_tfidf)
        return normalize(q_emb, norm="l2")[0].astype(np.float32)

    def _expand_parent(self, chunk: ChunkRecord) -> str:
        if chunk.parent_id and chunk.parent_id in self.parent_map:
            return self.parent_map[chunk.parent_id].text
        return ""

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[ScoredChunk]:
        scores = self.bm25.get_scores(tokenize_id(query))
        ranked_idx = np.argsort(scores)[::-1]
        results: list[ScoredChunk] = []
        for idx in ranked_idx:
            ch = self.chunks[int(idx)]
            if not _matches_filters(ch, filters):
                continue
            results.append(
                ScoredChunk(
                    chunk=ch,
                    score=float(scores[int(idx)]),
                    source_retriever="bm25",
                    parent_text=self._expand_parent(ch),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[ScoredChunk]:
        q_vec = self._encode_query(query)
        scores = self.dense_matrix @ q_vec
        ranked_idx = np.argsort(scores)[::-1]
        results: list[ScoredChunk] = []
        for idx in ranked_idx:
            ch = self.chunks[int(idx)]
            if not _matches_filters(ch, filters):
                continue
            results.append(
                ScoredChunk(
                    chunk=ch,
                    score=float(scores[int(idx)]),
                    source_retriever="dense",
                    parent_text=self._expand_parent(ch),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
        rrf_k: int = 60,
        candidate_pool: int = 40,
    ) -> list[ScoredChunk]:
        """Fuse BM25 and Dense via Reciprocal Rank Fusion (RRF)."""
        bm25_hits = self.search_bm25(query, top_k=candidate_pool, filters=filters)
        dense_hits = self.search_dense(query, top_k=candidate_pool, filters=filters)

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_by_id: dict[str, ScoredChunk] = {}

        for rank, item in enumerate(bm25_hits, start=1):
            cid = item.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (rrf_k + rank)
            chunk_by_id[cid] = item

        for rank, item in enumerate(dense_hits, start=1):
            cid = item.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (rrf_k + rank)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = item

        ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            ScoredChunk(
                chunk=chunk_by_id[cid].chunk,
                score=float(score),
                source_retriever="hybrid_rrf",
                parent_text=chunk_by_id[cid].parent_text,
            )
            for cid, score in ordered
        ]

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 10) -> list[ScoredChunk]:
        """Rerank fused candidates using CrossEncoder or bigram/pasal-aware lexical cross-scorer."""
        if not candidates:
            return []

        if self._cross_encoder is not None:
            pairs = [(query, c.chunk.text[:1500]) for c in candidates]
            scores = self._cross_encoder.predict(pairs)
            rescored = [
                ScoredChunk(
                    chunk=c.chunk,
                    score=float(s),
                    source_retriever="hybrid_rerank",
                    parent_text=c.parent_text,
                )
                for c, s in zip(candidates, scores)
            ]
            return sorted(rescored, key=lambda x: x.score, reverse=True)[:top_k]

        # Deterministic CPU lexical+bigram+pasal-mention cross-scorer
        q_tokens = set(tokenize_id(query))
        q_bigrams = set(zip(tokenize_id(query), tokenize_id(query)[1:]))
        rescored: list[ScoredChunk] = []
        for rank_idx, item in enumerate(candidates):
            doc_text = f"{item.chunk.title} {item.chunk.section_path} {item.chunk.text[:1800]}"
            d_tokens = tokenize_id(doc_text)
            d_set = set(d_tokens)
            d_bigrams = set(zip(d_tokens, d_tokens[1:]))

            uni_overlap = len(q_tokens & d_set) / max(1, len(q_tokens))
            bi_overlap = len(q_bigrams & d_bigrams) / max(1, len(q_bigrams)) if q_bigrams else 0.0
            pasal_bonus = 0.15 if (item.chunk.pasal and item.chunk.pasal.lower() in query.lower()) else 0.0
            rrf_prior = 1.0 / (10.0 + rank_idx)
            final_score = 0.55 * uni_overlap + 0.30 * bi_overlap + pasal_bonus + 0.15 * rrf_prior

            rescored.append(
                ScoredChunk(
                    chunk=item.chunk,
                    score=round(float(final_score), 6),
                    source_retriever="hybrid_rerank",
                    parent_text=item.parent_text,
                )
            )
        return sorted(rescored, key=lambda x: x.score, reverse=True)[:top_k]

    def retrieve(
        self,
        query: str,
        mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[ScoredChunk]:
        if mode == "bm25":
            return self.search_bm25(query, top_k=top_k, filters=filters)
        if mode == "dense":
            return self.search_dense(query, top_k=top_k, filters=filters)
        if mode == "hybrid":
            return self.search_hybrid(query, top_k=top_k, filters=filters)
        if mode == "hybrid_rerank":
            fused = self.search_hybrid(query, top_k=max(top_k * 3, 25), filters=filters)
            return self.rerank(query, fused, top_k=top_k)
        raise ValueError(f"Unsupported retrieval mode: {mode}")


def evaluate_retrieval(
    index: PolicyRetrievalIndex,
    queries: list[dict[str, Any]],
    qrels: list[QrelEntry],
    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"],
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute Recall@5, Recall@10, nDCG@10, MRR@10, and latency on identical qrels."""
    qrel_by_qid: dict[str, dict[str, int]] = defaultdict(dict)
    qrel_pasals_by_qid: dict[str, set[tuple[str, str]]] = defaultdict(set)
    qrel_docs_by_qid: dict[str, set[str]] = defaultdict(set)
    all_qrel_chunk_ids: set[str] = set()
    for qr in qrels:
        if qr.relevance > 0:
            all_qrel_chunk_ids.add(qr.chunk_id)
            qrel_by_qid[qr.query_id][qr.chunk_id] = max(qrel_by_qid[qr.query_id].get(qr.chunk_id, 0), qr.relevance)
            qrel_docs_by_qid[qr.query_id].add(qr.doc_id)
            if qr.pasal:
                qrel_pasals_by_qid[qr.query_id].add((qr.doc_id, qr.pasal.lower()))

    index_chunk_ids = {c.chunk_id for c in index.chunks}
    match_by_exact_chunk = bool(index_chunk_ids & all_qrel_chunk_ids)

    recall5_list: list[float] = []
    recall10_list: list[float] = []
    ndcg10_list: list[float] = []
    mrr10_list: list[float] = []
    latencies_ms: list[float] = []

    for q in queries:
        qid = q["query_id"]
        rel_chunks = qrel_by_qid.get(qid, {})
        rel_pasals = qrel_pasals_by_qid.get(qid, set())
        rel_docs = qrel_docs_by_qid.get(qid, set())
        if not rel_chunks and not rel_docs:
            continue

        t0 = time.perf_counter()
        hits = index.retrieve(q["query"], mode=mode, top_k=max(10, top_k))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        gains: list[int] = []
        matched_targets: set[Any] = set()

        def _hit_target(h: ScoredChunk) -> tuple[Any, int]:
            if match_by_exact_chunk:
                g = rel_chunks.get(h.chunk.chunk_id, 0)
                return (h.chunk.chunk_id if g > 0 else None), g
            # Cross-chunker comparison (Ablation A1): match exact (doc_id, pasal) when pasal is known
            if rel_pasals:
                key = (h.chunk.doc_id, (h.chunk.pasal or "").lower())
                if key in rel_pasals:
                    return key, 2
                # Fallback check if pasal text appears inside fixed chunk window
                for d_id, p_str in rel_pasals:
                    if h.chunk.doc_id == d_id and p_str in h.chunk.text.lower():
                        return (d_id, p_str), 1
                return None, 0
            if h.chunk.doc_id in rel_docs:
                return h.chunk.doc_id, 1
            return None, 0

        for h in hits[:10]:
            tgt, g = _hit_target(h)
            if tgt is not None and tgt not in matched_targets:
                gains.append(g)
                matched_targets.add(tgt)
            else:
                gains.append(0)

        if match_by_exact_chunk:
            denom = max(1, len(rel_chunks))
        elif rel_pasals:
            denom = max(1, len(rel_pasals))
        else:
            denom = max(1, len(rel_docs))

        hits_5 = set()
        for h in hits[:5]:
            tgt, _ = _hit_target(h)
            if tgt is not None:
                hits_5.add(tgt)

        recall5_list.append(min(1.0, len(hits_5) / denom))
        recall10_list.append(min(1.0, len(matched_targets) / denom))

        rr = 0.0
        for rank, g in enumerate(gains, start=1):
            if g > 0:
                rr = 1.0 / rank
                break
        mrr10_list.append(rr)

        dcg = sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains, start=1) if g > 0)
        ideal_gains = sorted(rel_chunks.values(), reverse=True)[:10] if match_by_exact_chunk else [2 if rel_pasals else 1] * min(10, denom)
        idcg = sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(ideal_gains, start=1))
        ndcg10_list.append(dcg / idcg if idcg > 0 else 0.0)

    lat_sorted = sorted(latencies_ms) if latencies_ms else [0.0]
    p95_idx = min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))

    return {
        **index.retrieval_metadata(),
        "mode": mode,
        "dense_backend": index.dense_backend,
        "reranker_backend": index.reranker_backend,
        "evaluated_queries": len(recall10_list),
        "Recall@5": round(float(np.mean(recall5_list)) if recall5_list else 0.0, 4),
        "Recall@10": round(float(np.mean(recall10_list)) if recall10_list else 0.0, 4),
        "nDCG@10": round(float(np.mean(ndcg10_list)) if ndcg10_list else 0.0, 4),
        "MRR@10": round(float(np.mean(mrr10_list)) if mrr10_list else 0.0, 4),
        "avg_latency_ms": round(float(np.mean(lat_sorted)), 2),
        "p95_latency_ms": round(float(lat_sorted[p95_idx]), 2),
    }
