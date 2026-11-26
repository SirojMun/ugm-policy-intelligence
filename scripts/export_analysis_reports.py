#!/usr/bin/env python3
"""Export structured CSV tables and final Markdown audit report from experiment JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _benchmark_audit(benchmark_dir: Path = Path("data/benchmark")) -> dict[str, Any]:
    manifest = _load_json(benchmark_dir / "split_manifest.json") or {}
    docs_by_split: dict[str, set[str]] = {}
    for split, fname in (("train", "train.jsonl"), ("dev", "dev.jsonl"), ("test", "test_locked.jsonl")):
        fp = benchmark_dir / fname
        s: set[str] = set()
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    s.update(json.loads(line).get("gold_doc_ids", []))
        docs_by_split[split] = s

    return {
        "counts_by_split": manifest.get("counts_by_split", {}),
        "total_qrels": manifest.get("total_qrels", 0),
        "leakage_and_dedup_check": manifest.get("leakage_and_dedup_check", {}),
        "unique_gold_docs": {k: len(v) for k, v in docs_by_split.items()},
        "doc_overlap": {
            "train_test": len(docs_by_split.get("train", set()) & docs_by_split.get("test", set())),
            "dev_test": len(docs_by_split.get("dev", set()) & docs_by_split.get("test", set())),
            "train_dev": len(docs_by_split.get("train", set()) & docs_by_split.get("dev", set())),
        },
    }


def export_analysis(report_dir: str | Path, out_subdir: str = "analysis") -> Path:
    """Convert raw summary JSON artifacts in `report_dir` into CSV tables and final_report.md."""
    base = Path(report_dir)
    out = base / out_subdir
    out.mkdir(parents=True, exist_ok=True)

    b0_b6 = _load_json(base / "b0_b6_summary.json") or {}
    abl = _load_json(base / "ablations_a1_a7.json") or {}
    rt = _load_json(base / "red_team_report.json") or {}
    qlora = _load_json(base / "qlora_training_metrics.json") or {}
    audit = _benchmark_audit()

    # 1. B0-B6 comparison CSV + per-slice CSV
    comp_rows = []
    slice_rows = []
    for cfg_id in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        if cfg_id not in b0_b6:
            continue
        m = b0_b6[cfg_id]
        comp_rows.append(
            {
                "config_id": cfg_id,
                "mean_correctness": m["mean_correctness"],
                "std_correctness": m["std_correctness"],
                "mean_verdict_acc": m["mean_verdict_acc"],
                "mean_evidence_recall": m["mean_evidence_recall"],
                "mean_citation_precision": m["mean_citation_precision"],
                "mean_temporal_acc": m["mean_temporal_acc"],
                "mean_groundedness": m["mean_groundedness"],
                "mean_false_confident": m["mean_false_confident"],
                "mean_latency_ms": m["mean_latency_ms"],
                "p95_latency_ms": m["p95_latency_ms"],
                "b6_parent_config": m.get("b6_parent_config") or "",
            }
        )
        for sl, sm in m.get("by_slice", {}).items():
            slice_rows.append({"config_id": cfg_id, "slice_type": sl, **sm})

    if comp_rows:
        with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comp_rows[0].keys()))
            w.writeheader()
            w.writerows(comp_rows)

    if slice_rows:
        with (out / "comparison_by_slice.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(slice_rows[0].keys()))
            w.writeheader()
            w.writerows(slice_rows)

    # 2. Ablations A1-A7 CSV
    abl_rows: list[dict[str, Any]] = []
    for abl_name, variants in abl.items():
        if not isinstance(variants, dict):
            continue
        for var_name, metrics in variants.items():
            if not isinstance(metrics, dict):
                continue
            flat = metrics.get("overall", metrics)
            if isinstance(flat, dict):
                row = {"ablation": abl_name, "variant": var_name}
                for k, v in flat.items():
                    if isinstance(v, (int, float, str, bool)):
                        row[k] = v
                abl_rows.append(row)

    if abl_rows:
        fieldnames = list(dict.fromkeys(k for r in abl_rows for k in r.keys()))
        with (out / "ablation.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(abl_rows)

    # 3. Red-Team CSV
    rt_cases = rt.get("cases", [])
    if rt_cases:
        with (out / "red_team.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rt_cases[0].keys()))
            w.writeheader()
            w.writerows(rt_cases)

    # 4. Markdown Audit Report
    lines = [
        f"# Academic Policy Intelligence — Experiment & Audit Report (`{base.name}`)",
        "",
        "## 1. Benchmark & Leakage Audit",
        f"- Splits: `{audit['counts_by_split']}`, Qrels: `{audit['total_qrels']}`",
        f"- Query-level dedup (`split_manifest.json`): `{audit['leakage_and_dedup_check']}`",
        f"- Unique gold documents by split: `{audit['unique_gold_docs']}`, cross-split gold doc overlap: `{audit['doc_overlap']}`",
        "",
    ]

    if qlora:
        lines.extend(
            [
                "## 2. QLoRA Fine-Tuning Summary",
                f"- Base model: `{qlora.get('base_model', '')}`",
                f"- Metrics: `{qlora.get('metrics', {})}`",
                "",
            ]
        )

    if comp_rows:
        lines.extend(
            [
                "## 3. B0–B6 Comparison on Locked Test (Mean ± Std across Independent Runs)",
                "",
                "| Config | Correctness | Verdict Acc | Evidence Recall | Citation Prec | Temporal Acc | Groundedness | False Confident | Mean Latency (ms) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for r in comp_rows:
            lines.append(
                f"| {r['config_id']} | {r['mean_correctness']} ± {r['std_correctness']} | {r['mean_verdict_acc']} | "
                f"{r['mean_evidence_recall']} | {r['mean_citation_precision']} | {r['mean_temporal_acc']} | "
                f"{r['mean_groundedness']} | {r['mean_false_confident']} | {r['mean_latency_ms']} |"
            )
        lines.append("")

    if rt:
        cat_counts = Counter()
        cat_pass = Counter()
        for c in rt_cases:
            cat_counts[c["category"]] += 1
            if c.get("mitigation_passed_defense"):
                cat_pass[c["category"]] += 1
        lines.extend(
            [
                "## 4. Red-Team Security Evaluation",
                f"- Total fixtures: `{rt.get('total_fixtures', 0)}`",
                f"- Baseline pass rate (unguarded B2): `{rt.get('baseline_pass_rate', 0.0)}`",
                f"- Mitigated pass rate (guarded B6): `{rt.get('mitigated_pass_rate', 0.0)}`",
                f"- Category pass counts (mitigated): `{ {k: f'{cat_pass[k]}/{cat_counts[k]}' for k in sorted(cat_counts)} }`",
                "",
            ]
        )

    (out / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSV & Markdown analysis tables from reports.")
    parser.add_argument("--report-dir", default="reports_kaggle")
    args = parser.parse_args()
    out = export_analysis(args.report_dir)
    print(f"Exported analysis tables & Markdown summary to: {out}")


if __name__ == "__main__":
    main()
