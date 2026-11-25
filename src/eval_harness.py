"""End-to-end evaluation harness for B0-B6 comparison, 3-run repeatability, and Ablations A1-A7."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.agents_pipeline import PolicyExperimentRunner
from src.chunking import chunk_corpus, compute_chunk_stats, load_corpus
from src.retrieval import PolicyRetrievalIndex, evaluate_retrieval, tokenize_id
from src.schemas import PolicyResponse, QrelEntry


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def score_response(item: dict[str, Any], resp: PolicyResponse) -> dict[str, Any]:
    """Compute answerability/verdict accuracy, evidence recall, citation precision, temporal & hallucination metrics."""
    gold_verdict = item["gold_verdict"]
    gold_docs = set(item.get("gold_doc_ids", []))
    gold_valid_time = str(item.get("gold_valid_time") or "")

    verdict_correct = float(resp.verdict == gold_verdict)
    cited_docs = {c.doc_id for c in resp.citations}

    if item["answerability"] == "answerable":
        evidence_recall = len(cited_docs & gold_docs) / max(1, len(gold_docs))
        citation_precision = len(cited_docs & gold_docs) / max(1, len(cited_docs)) if cited_docs else 0.0
        temporal_ok = float(
            bool(gold_valid_time) and (gold_valid_time[:4] in str(resp.valid_time) or gold_valid_time[:4] in resp.answer)
        )
        # Answer lexical/keyword grounding vs gold_answer
        gold_toks = {t for t in tokenize_id(item.get("gold_answer", "")) if len(t) >= 4}
        ans_toks = set(tokenize_id(resp.answer))
        answer_f1_proxy = len(gold_toks & ans_toks) / max(1, len(gold_toks)) if gold_toks else verdict_correct
        false_refusal = 1.0 if resp.verdict == "unknown" else 0.0
        false_confident = 0.0
    else:
        # Unanswerable: ideal behavior is abstain (verdict == "unknown")
        evidence_recall = 1.0 if resp.verdict == "unknown" else 0.0
        citation_precision = 1.0 if resp.verdict == "unknown" else 0.0
        temporal_ok = 1.0 if resp.verdict == "unknown" else 0.0
        answer_f1_proxy = 1.0 if resp.verdict == "unknown" else 0.0
        false_refusal = 0.0
        false_confident = 1.0 if resp.verdict == "supported" else 0.0

    total_claims = max(1, len(resp.atomic_claims))
    unsupported_claims = sum(1 for c in resp.atomic_claims if c.verdict != "supported")
    unsupported_rate = unsupported_claims / total_claims if resp.atomic_claims else (0.0 if resp.verdict == "unknown" else 1.0)
    groundedness = 1.0 - unsupported_rate if resp.verdict == "supported" else (1.0 if item["answerability"] == "unanswerable" else 0.0)

    # Composite correctness: verdict + evidence + groundedness
    correctness = round(
        0.4 * verdict_correct + 0.3 * answer_f1_proxy + 0.3 * evidence_recall,
        4,
    )

    return {
        "query_id": item["query_id"],
        "slice_type": item["slice_type"],
        "verdict_correct": verdict_correct,
        "correctness": correctness,
        "evidence_recall": round(evidence_recall, 4),
        "citation_precision": round(citation_precision, 4),
        "temporal_accuracy": round(temporal_ok, 4),
        "groundedness": round(groundedness, 4),
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "false_refusal": false_refusal,
        "false_confident": false_confident,
        "latency_ms": resp.latency_ms,
        "total_tokens_est": resp.prompt_tokens_est + resp.completion_tokens_est,
    }


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    lats = sorted(r["latency_ms"] for r in rows)
    p95 = lats[min(len(lats) - 1, int(0.95 * len(lats)))]
    keys = [
        "verdict_correct",
        "correctness",
        "evidence_recall",
        "citation_precision",
        "temporal_accuracy",
        "groundedness",
        "unsupported_claim_rate",
        "false_refusal",
        "false_confident",
        "total_tokens_est",
    ]
    summary = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    summary["n"] = len(rows)
    summary["avg_latency_ms"] = round(float(np.mean(lats)), 2)
    summary["p95_latency_ms"] = round(float(p95), 2)
    return summary


def run_b0_b6_suite(
    runner: PolicyExperimentRunner,
    split_rows: list[dict[str, Any]],
    seeds: tuple[int, ...] = (101, 202, 303),
    out_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Run B0-B6 across >=3 independent runs on test split and compute mean/std variance."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    configs = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
    results_summary: dict[str, Any] = {}
    run_manifests: list[dict[str, Any]] = []

    raw_outputs_path = out_path / "b0_b6_raw_outputs.jsonl"
    with raw_outputs_path.open("w", encoding="utf-8") as raw_fh:
        for cfg_id in configs:
            per_seed_summaries: list[dict[str, Any]] = []
            slice_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

            for seed in seeds:
                seed_rows: list[dict[str, Any]] = []
                total_q = len(split_rows)
                for q_idx, item in enumerate(split_rows, start=1):
                    resp = runner.run(
                        config_id=cfg_id,  # type: ignore[arg-type]
                        query=item["query"],
                        query_id=item["query_id"],
                        as_of_date=item.get("as_of_date", "2026-11-25"),
                        seed=seed,
                    )
                    if q_idx % 25 == 0 or q_idx == total_q:
                        print(f"[{cfg_id} seed={seed}] processed {q_idx}/{total_q}", flush=True)
                    scored = score_response(item, resp)
                    seed_rows.append(scored)
                    if seed == seeds[0]:
                        slice_buckets[item["slice_type"]].append(scored)

                    raw_fh.write(
                        json.dumps(
                            {
                                "config_id": cfg_id,
                                "seed": seed,
                                "query_id": item["query_id"],
                                "slice_type": item["slice_type"],
                                "response": resp.model_dump(),
                                "metrics": scored,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                agg = aggregate_scores(seed_rows)
                per_seed_summaries.append(agg)
                run_manifests.append(
                    {
                        "config_id": cfg_id,
                        "seed": seed,
                        "model_id": (
                            runner.peft_backend.model_id
                            if (cfg_id == "B5" or (cfg_id == "B6" and runner.b6_parent_config == "B5"))
                            else runner.base_backend.model_id
                        ),
                        "b6_parent_config": runner.b6_parent_config if cfg_id == "B6" else None,
                        "index_version": runner.index_version,
                        "metrics": agg,
                    }
                )

            results_summary[cfg_id] = {
                "mean_correctness": round(float(np.mean([s["correctness"] for s in per_seed_summaries])), 4),
                "std_correctness": round(float(np.std([s["correctness"] for s in per_seed_summaries])), 4),
                "mean_verdict_acc": round(float(np.mean([s["verdict_correct"] for s in per_seed_summaries])), 4),
                "mean_evidence_recall": round(float(np.mean([s["evidence_recall"] for s in per_seed_summaries])), 4),
                "mean_citation_precision": round(float(np.mean([s["citation_precision"] for s in per_seed_summaries])), 4),
                "mean_temporal_acc": round(float(np.mean([s["temporal_accuracy"] for s in per_seed_summaries])), 4),
                "mean_groundedness": round(float(np.mean([s["groundedness"] for s in per_seed_summaries])), 4),
                "mean_false_confident": round(float(np.mean([s["false_confident"] for s in per_seed_summaries])), 4),
                "mean_latency_ms": round(float(np.mean([s["avg_latency_ms"] for s in per_seed_summaries])), 2),
                "p95_latency_ms": round(float(np.mean([s["p95_latency_ms"] for s in per_seed_summaries])), 2),
                "b6_parent_config": runner.b6_parent_config if cfg_id == "B6" else None,
                "by_slice": {sl: aggregate_scores(rows) for sl, rows in slice_buckets.items()},
            }

    (out_path / "run_manifests.json").write_text(json.dumps(run_manifests, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_path / "b0_b6_summary.json").write_text(json.dumps(results_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results_summary


def run_ablations_a1_a7(
    corpus_jsonl: str | Path = "corpus_clean.jsonl",
    dev_jsonl: str | Path = "data/benchmark/dev.jsonl",
    qrels_jsonl: str | Path = "data/benchmark/qrels.jsonl",
    train_jsonl: str | Path = "data/benchmark/train.jsonl",
    out_dir: str | Path = "reports",
    runner: PolicyExperimentRunner | None = None,
    fixed_chunks: list[Any] | None = None,
    head_chunks: list[Any] | None = None,
    hier_chunks: list[Any] | None = None,
) -> dict[str, Any]:
    """Execute all 7 single-variable ablations (A1-A7) on the dev split."""
    docs = load_corpus(corpus_jsonl)
    docs_by_id = {d["doc_id"]: d for d in docs}
    dev_rows = load_jsonl_rows(dev_jsonl)
    dev_answerable = [r for r in dev_rows if r["answerability"] == "answerable"]
    train_rows = load_jsonl_rows(train_jsonl)
    train_examples = [{"query": r["query"], "answer": r["gold_answer"]} for r in train_rows if r["answerability"] == "answerable"][:3]

    qrels = [QrelEntry(**r) for r in load_jsonl_rows(qrels_jsonl)]

    if runner is not None:
        idx_fixed = runner.fixed_index
        idx_head = runner.heading_index
        idx_hier = runner.hierarchical_index
        fixed_chunks = fixed_chunks or idx_fixed.chunks
        head_chunks = head_chunks or idx_head.chunks
        hier_chunks = hier_chunks or (list(idx_hier.parent_map.values()) + idx_hier.chunks)
    else:
        fixed_chunks = fixed_chunks or chunk_corpus(docs, "fixed")
        head_chunks = head_chunks or chunk_corpus(docs, "heading_aware")
        hier_chunks = hier_chunks or chunk_corpus(docs, "hierarchical")
        idx_fixed = PolicyRetrievalIndex(fixed_chunks)
        idx_head = PolicyRetrievalIndex(head_chunks)
        idx_hier = PolicyRetrievalIndex(hier_chunks)
        runner = PolicyExperimentRunner(
            docs_by_id=docs_by_id,
            fixed_index=idx_fixed,
            heading_index=idx_head,
            hierarchical_index=idx_hier,
            train_examples=train_examples,
        )

    # A1: fixed vs heading_aware vs hierarchical chunking (uses same runner backends)
    a1 = {}
    for name, ch_list, idx in (
        ("fixed", fixed_chunks, idx_fixed),
        ("heading_aware", head_chunks, idx_head),
        ("hierarchical", hier_chunks, idx_hier),
    ):
        ret_m = evaluate_retrieval(idx, dev_answerable, qrels, mode="hybrid_rerank", top_k=10)
        stats = compute_chunk_stats(ch_list)
        a1_runner = PolicyExperimentRunner(
            docs_by_id=docs_by_id,
            fixed_index=idx,
            heading_index=idx,
            hierarchical_index=idx,
            base_backend=runner.base_backend,
            peft_backend=runner.peft_backend,
            train_examples=train_examples,
            index_version=f"a1-{name}",
        )
        b_rows = [score_response(item, a1_runner.run("B4", item["query"], item["query_id"])) for item in dev_rows]
        a1[name] = {**stats, **ret_m, "groundedness": aggregate_scores(b_rows)["groundedness"]}

    # A2: dense vs bm25 vs hybrid vs hybrid_rerank on identical hierarchical qrels
    a2 = {
        mode: evaluate_retrieval(idx_hier, dev_answerable, qrels, mode=mode, top_k=10)  # type: ignore[arg-type]
        for mode in ("dense", "bm25", "hybrid", "hybrid_rerank")
    }

    # A3: no-verifier vs verifier + bounded repair on B4
    a3 = {}
    for label, use_ver in (("no_verifier", False), ("verifier_bounded_repair", True)):
        rows = [score_response(it, runner.run("B4", it["query"], it["query_id"], enable_verifier=use_ver)) for it in dev_rows]
        a3[label] = aggregate_scores(rows)

    # A4: no-adapter (B4) vs LoRA/QLoRA (B5), including train-vs-dev overfit gap & OOD slice
    b4_dev_rows = [score_response(it, runner.run("B4", it["query"], it["query_id"])) for it in dev_rows]
    b5_dev_rows = [score_response(it, runner.run("B5", it["query"], it["query_id"])) for it in dev_rows]
    train_subset = train_rows[:15]
    b5_train_rows = [score_response(it, runner.run("B5", it["query"], it["query_id"])) for it in train_subset]
    ood_dev_rows_b4 = [r for r in b4_dev_rows if r["slice_type"] == "no-evidence"]
    ood_dev_rows_b5 = [r for r in b5_dev_rows if r["slice_type"] == "no-evidence"]

    b4_dev_agg = aggregate_scores(b4_dev_rows)
    b5_dev_agg = aggregate_scores(b5_dev_rows)
    b5_train_agg = aggregate_scores(b5_train_rows)
    selected_b6_parent = "B5" if b5_dev_agg["correctness"] >= b4_dev_agg["correctness"] else "B4"

    a4 = {
        "B4_no_adapter": b4_dev_agg,
        "B5_qlora_adapter": b5_dev_agg,
        "B5_train_subset_check": b5_train_agg,
        "overfit_gap_correctness": round(b5_train_agg.get("correctness", 0.0) - b5_dev_agg.get("correctness", 0.0), 4),
        "OOD_no_evidence_B4": aggregate_scores(ood_dev_rows_b4),
        "OOD_no_evidence_B5": aggregate_scores(ood_dev_rows_b5),
        "selected_b6_parent": selected_b6_parent,
    }

    # A5: B6 without vs with tool calling
    a5 = {
        "B6_without_tools": aggregate_scores([score_response(it, runner.run("B6", it["query"], it["query_id"], enable_tools=False)) for it in dev_rows]),
        "B6_with_guarded_tools": aggregate_scores([score_response(it, runner.run("B6", it["query"], it["query_id"], enable_tools=True)) for it in dev_rows]),
    }

    # A6: fixed vs adaptive retrieval policy across slices
    a6 = {}
    for label, adapt in (("fixed_retrieval", False), ("adaptive_retrieval", True)):
        rows = [score_response(it, runner.run("B4", it["query"], it["query_id"], adaptive_retrieval=adapt)) for it in dev_rows]
        by_sl: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_sl[r["slice_type"]].append(r)
        a6[label] = {
            "overall": aggregate_scores(rows),
            "by_slice": {sl: aggregate_scores(rs) for sl, rs in by_sl.items()},
        }

    # A7: 2 token/context budgets (2000 vs 6000)
    a7 = {
        f"budget_{b}": aggregate_scores(
            [score_response(it, runner.run("B4", it["query"], it["query_id"], max_context_tokens=b)) for it in dev_rows]
        )
        for b in (2000, 6000)
    }

    ablations = {
        "A1_chunking": a1,
        "A2_retrieval": a2,
        "A3_verifier": a3,
        "A4_peft_adapter": a4,
        "A5_agentic_tools": a5,
        "A6_retrieval_policy": a6,
        "A7_context_budget": a7,
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "ablations_a1_a7.json").write_text(json.dumps(ablations, indent=2, ensure_ascii=False), encoding="utf-8")
    return ablations
