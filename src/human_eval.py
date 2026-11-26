"""Blind dual-annotator export, inter-annotator agreement (Cohen's kappa), and adjudication workflow."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.retrieval import PolicyRetrievalIndex

RELEVANCE_RUBRIC = {
    "0": "tidak_relevan",
    "1": "relevansi_parsial",
    "2": "relevan",
    "3": "sangat_relevan_menjawab_langsung",
}


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute unweighted Cohen's kappa between two categorical annotation sequences."""
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    n = len(labels_a)
    classes = sorted(set(labels_a) | set(labels_b))
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    p_e = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in classes)
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def export_blind_annotation_packets(
    queries_jsonl: str | Path,
    index: PolicyRetrievalIndex,
    out_dir: str | Path = "data/annotation",
    top_k_candidates: int = 3,
    random_samples_per_query: int = 2,
    seed: int = 42,
) -> dict[str, str]:
    """Export blind annotation worksheets using multi-retriever pooling + random negatives (0-3 rubric)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    queries = [json.loads(l) for l in Path(queries_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    blind_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    all_chunks = index.chunks

    for q in queries:
        pooled_by_cid: dict[str, Any] = {}
        for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
            for h in index.retrieve(q["query"], mode=mode, top_k=top_k_candidates):  # type: ignore[arg-type]
                if h.chunk.chunk_id not in pooled_by_cid:
                    pooled_by_cid[h.chunk.chunk_id] = h.chunk

        # Add unretrieved corpus samples so annotators are not limited to top-k system biases
        if all_chunks and random_samples_per_query > 0:
            rand_pool = rng.sample(all_chunks, k=min(len(all_chunks), random_samples_per_query * 2))
            added = 0
            for ch in rand_pool:
                if ch.chunk_id not in pooled_by_cid:
                    pooled_by_cid[ch.chunk_id] = ch
                    added += 1
                    if added >= random_samples_per_query:
                        break

        pooled_chunks = list(pooled_by_cid.values())
        rng.shuffle(pooled_chunks)  # Blind order hides which retriever surfaced each passage
        candidates = [
            {
                "doc_id": ch.doc_id,
                "chunk_id": ch.chunk_id,
                "pasal": ch.pasal,
                "valid_time": ch.valid_time,
                "status": ch.status,
                "excerpt": ch.text[:450],
            }
            for ch in pooled_chunks
        ]
        blind_rows.append(
            {
                "query_id": q["query_id"],
                "query": q["query"],
                "as_of_date": q.get("as_of_date", "2026-11-25"),
                "relevance_rubric": RELEVANCE_RUBRIC,
                "candidate_passages": candidates,
                "annotator_verdict": "",  # supported | unknown
                "annotator_relevant_doc_ids": [],
                "annotator_relevance_scores": {},  # optional chunk_id -> 0..3
                "annotator_valid_time": "",
                "annotator_notes": "",
            }
        )

    p1 = out_path / "annotator_1_blind.jsonl"
    p2 = out_path / "annotator_2_blind.jsonl"
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in blind_rows) + "\n"
    p1.write_text(payload, encoding="utf-8")
    p2.write_text(payload, encoding="utf-8")
    return {"annotator_1": str(p1), "annotator_2": str(p2), "count": str(len(blind_rows))}


def compute_agreement_and_adjudicate(
    annotator_1_jsonl: str | Path,
    annotator_2_jsonl: str | Path,
    adjudication_jsonl: str | Path | None = None,
    out_dir: str | Path = "data/annotation",
) -> dict[str, Any]:
    """Compare dual blind annotations, emit disagreement log, and merge adjudications."""
    r1 = {r["query_id"]: r for r in (json.loads(l) for l in Path(annotator_1_jsonl).read_text(encoding="utf-8").splitlines() if l.strip())}
    r2 = {r["query_id"]: r for r in (json.loads(l) for l in Path(annotator_2_jsonl).read_text(encoding="utf-8").splitlines() if l.strip())}
    adj = {}
    if adjudication_jsonl and Path(adjudication_jsonl).exists():
        adj = {r["query_id"]: r for r in (json.loads(l) for l in Path(adjudication_jsonl).read_text(encoding="utf-8").splitlines() if l.strip())}

    common_ids = sorted(set(r1.keys()) & set(r2.keys()))
    verdicts_1 = [r1[qid].get("annotator_verdict", "unknown") for qid in common_ids]
    verdicts_2 = [r2[qid].get("annotator_verdict", "unknown") for qid in common_ids]

    disagreements: list[dict[str, Any]] = []
    adjudicated_records: list[dict[str, Any]] = []

    for qid in common_ids:
        a1, a2 = r1[qid], r2[qid]
        v1 = a1.get("annotator_verdict", "unknown")
        v2 = a2.get("annotator_verdict", "unknown")
        docs1 = sorted(set(a1.get("annotator_relevant_doc_ids", [])))
        docs2 = sorted(set(a2.get("annotator_relevant_doc_ids", [])))
        has_conflict = (v1 != v2) or (docs1 != docs2)

        resolution = adj.get(qid, {})
        if has_conflict:
            disagreements.append(
                {
                    "query_id": qid,
                    "query": a1.get("query", ""),
                    "annotator_1_verdict": v1,
                    "annotator_2_verdict": v2,
                    "annotator_1_docs": docs1,
                    "annotator_2_docs": docs2,
                    "resolved_verdict": resolution.get("resolved_verdict", ""),
                    "resolved_docs": resolution.get("resolved_docs", []),
                    "resolution_notes": resolution.get("resolution_notes", "PENDING_ADJUDICATION"),
                }
            )

        resolved_verdict = (
            resolution["resolved_verdict"]
            if "resolved_verdict" in resolution and resolution["resolved_verdict"]
            else (v1 if v1 == v2 else "unknown")
        )
        resolved_docs = (
            resolution["resolved_docs"]
            if "resolved_docs" in resolution
            else (docs1 if docs1 == docs2 else sorted(set(docs1) | set(docs2)))
        )
        resolved_valid_time = (
            resolution["resolved_valid_time"]
            if "resolved_valid_time" in resolution
            else (a1.get("annotator_valid_time") or a2.get("annotator_valid_time", ""))
        )
        adjudicated_records.append(
            {
                "query_id": qid,
                "verdict": resolved_verdict,
                "relevant_doc_ids": resolved_docs,
                "valid_time": resolved_valid_time,
                "adjudicated": has_conflict,
                "resolution_notes": resolution.get("resolution_notes", "unanimous_agreement" if not has_conflict else "PENDING_ADJUDICATION"),
            }
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    disagreement_path = out_path / "disagreement_log.jsonl"
    disagreement_path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in disagreements) + ("\n" if disagreements else ""),
        encoding="utf-8",
    )
    adjudicated_path = out_path / "adjudicated_labels.jsonl"
    adjudicated_path.write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in adjudicated_records) + ("\n" if adjudicated_records else ""),
        encoding="utf-8",
    )

    exact_agree = sum(1 for v1, v2 in zip(verdicts_1, verdicts_2) if v1 == v2)
    doc_agree = sum(
        1
        for qid in common_ids
        if set(r1[qid].get("annotator_relevant_doc_ids", [])) == set(r2[qid].get("annotator_relevant_doc_ids", []))
    )

    summary = {
        "total_paired": len(common_ids),
        "verdict_agreement_pct": round(100.0 * exact_agree / max(1, len(common_ids)), 2),
        "verdict_cohen_kappa": cohen_kappa(verdicts_1, verdicts_2),
        "doc_set_agreement_pct": round(100.0 * doc_agree / max(1, len(common_ids)), 2),
        "disagreement_count": len(disagreements),
        "disagreement_log": str(disagreement_path),
        "adjudicated_labels": str(adjudicated_path),
    }
    (out_path / "agreement_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
