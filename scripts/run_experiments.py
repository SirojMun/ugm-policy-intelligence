#!/usr/bin/env python3
"""Master experiment runner for B0-B6 (3 independent runs), Ablations A1-A7, and Red-Team suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents_pipeline import PolicyExperimentRunner
from src.chunking import chunk_corpus, load_corpus
from src.eval_harness import load_jsonl_rows, run_ablations_a1_a7, run_b0_b6_suite
from src.llm_backend import BaseLLMBackend, ExtractivePolicyBackend, HuggingFaceCausalBackend
from src.red_team_runner import is_mitigated_safely
from src.retrieval import PolicyRetrievalIndex


def run_red_team(runner: PolicyExperimentRunner, out_path: str | Path = "reports/red_team_report.json") -> dict:
    fixtures = load_jsonl_rows("red_team/red_team_suite.jsonl")
    records = []
    for fx in fixtures:
        tool_attempt = fx.get("tool_call_attempt")
        inj = fx.get("injected_doc_context", "")

        # Baseline (B0 / B2 unguarded)
        base_resp = runner.run("B2", fx["query"], fx["fixture_id"], extra_injected_context=inj)
        base_tool = {"allowed": True, "reason": "no_tool_guard_in_baseline"} if tool_attempt else None
        base_ok, base_reason = is_mitigated_safely(fx, base_resp, base_tool)

        # Mitigated (B6 Guarded Agentic)
        mit_resp = runner.run("B6", fx["query"], fx["fixture_id"], extra_injected_context=inj)
        mit_tool = (
            runner.tool_guard.validate_and_execute(tool_attempt["tool"], tool_attempt["args"], calls_so_far=0)
            if tool_attempt
            else None
        )
        mit_ok, mit_reason = is_mitigated_safely(fx, mit_resp, mit_tool)

        records.append(
            {
                "fixture_id": fx["fixture_id"],
                "category": fx["category"],
                "goal": fx["goal"],
                "attack_surface": fx["attack_surface"],
                "baseline_passed_defense": base_ok,
                "baseline_outcome": base_reason,
                "mitigation_passed_defense": mit_ok,
                "mitigation_outcome": mit_reason,
            }
        )

    summary = {
        "total_fixtures": len(records),
        "baseline_pass_rate": round(sum(1 for r in records if r["baseline_passed_defense"]) / max(1, len(records)), 4),
        "mitigated_pass_rate": round(sum(1 for r in records if r["mitigation_passed_defense"]) / max(1, len(records)), 4),
        "cases": records,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute B0-B6, A1-A7 ablations, and Red-Team evaluation.")
    parser.add_argument("--backend", choices=("extractive", "hf"), default="extractive")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--dense-model", default=None, help="SentenceTransformer model e.g. intfloat/multilingual-e5-small or BAAI/bge-m3")
    parser.add_argument("--reranker-model", default=None, help="CrossEncoder model e.g. BAAI/bge-reranker-base or BAAI/bge-reranker-v2-m3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    docs = load_corpus("corpus_clean.jsonl")
    docs_by_id = {d["doc_id"]: d for d in docs}
    train_rows = load_jsonl_rows("data/benchmark/train.jsonl")
    train_examples = [{"query": r["query"], "answer": r["gold_answer"]} for r in train_rows if r["answerability"] == "answerable"][:3]
    test_rows = load_jsonl_rows("data/benchmark/test_locked.jsonl")

    fixed_chunks = chunk_corpus(docs, "fixed")
    head_chunks = chunk_corpus(docs, "heading_aware")
    hier_chunks = chunk_corpus(docs, "hierarchical")

    idx_fixed = PolicyRetrievalIndex(fixed_chunks, dense_model_name=args.dense_model, reranker_model_name=args.reranker_model)
    idx_head = PolicyRetrievalIndex(head_chunks, dense_model_name=args.dense_model, reranker_model_name=args.reranker_model)
    idx_hier = PolicyRetrievalIndex(hier_chunks, dense_model_name=args.dense_model, reranker_model_name=args.reranker_model)

    if args.backend == "hf":
        base_be: BaseLLMBackend = HuggingFaceCausalBackend(args.base_model, adapter_dir=None)
        peft_be: BaseLLMBackend = (
            HuggingFaceCausalBackend(args.base_model, adapter_dir=args.adapter_dir) if args.adapter_dir else base_be
        )
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

    print("1/3 Running A1-A7 ablations on dev...")
    abl = run_ablations_a1_a7(
        out_dir=args.out_dir,
        runner=runner,
        fixed_chunks=fixed_chunks,
        head_chunks=head_chunks,
        hier_chunks=hier_chunks,
    )
    selected_parent = abl["A4_peft_adapter"]["selected_b6_parent"]
    runner.b6_parent_config = selected_parent
    print("Selected B6 parent on dev:", selected_parent)
    print("A2 retrieval:", json.dumps(abl["A2_retrieval"], indent=2))

    print("2/3 Running B0-B6 (3 independent runs) on locked test...")
    b_res = run_b0_b6_suite(runner, test_rows, seeds=tuple(args.seeds), out_dir=args.out_dir)
    print("B0-B6 summary:", json.dumps({k: {"correctness": v["mean_correctness"], "std": v["std_correctness"], "evidence_recall": v["mean_evidence_recall"], "false_confident": v["mean_false_confident"]} for k, v in b_res.items()}, indent=2))

    print("3/3 Running Red-Team suite (20 cases)...")
    rt = run_red_team(runner, out_path=Path(args.out_dir) / "red_team_report.json")
    print("Red-Team summary:", {"total": rt["total_fixtures"], "baseline_pass": rt["baseline_pass_rate"], "mitigated_pass": rt["mitigated_pass_rate"]})


if __name__ == "__main__":
    main()
