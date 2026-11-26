# Academic Policy Intelligence — Experiment & Audit Report (`reports`)

## 1. Benchmark & Leakage Audit
- Splits: `{'train': 60, 'dev': 30, 'test': 110}`, Qrels: `239`
- Query-level dedup (`split_manifest.json`): `{'total_queries': 200, 'exact_duplicate_count': 0, 'cross_split_near_dup_count': 0, 'near_dup_threshold': 0.72, 'near_dups_sample': [], 'leakage_free': True}`
- Unique gold documents by split: `{'train': 47, 'dev': 27, 'test': 87}`, cross-split gold doc overlap: `{'train_test': 2, 'dev_test': 4, 'train_dev': 1}`

## 3. B0–B6 Comparison on Locked Test (Mean ± Std across Independent Runs)

| Config | Correctness | Verdict Acc | Evidence Recall | Citation Prec | Temporal Acc | Groundedness | False Confident | Mean Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.2294 ± 0.0 | 0.2273 | 0.2273 | 0.2273 | 0.2273 | 0.2273 | 0.0 | 0.04 |
| B1 | 0.2365 ± 0.0 | 0.2273 | 0.2273 | 0.2273 | 0.2273 | 0.2273 | 0.0 | 0.06 |
| B2 | 0.6011 ± 0.0 | 0.7727 | 0.6136 | 0.3656 | 0.7182 | 1.0 | 0.2273 | 24.53 |
| B3 | 0.6127 ± 0.0 | 0.7727 | 0.6455 | 0.5556 | 0.7636 | 1.0 | 0.2273 | 226.83 |
| B4 | 0.7691 ± 0.0 | 0.9182 | 0.8364 | 0.7162 | 0.9091 | 1.0 | 0.0818 | 429.27 |
| B5 | 0.7691 ± 0.0 | 0.9182 | 0.8364 | 0.7162 | 0.9091 | 1.0 | 0.0818 | 322.96 |
| B6 | 0.7665 ± 0.0 | 0.9182 | 0.8273 | 0.7268 | 0.9091 | 1.0 | 0.0818 | 270.27 |

## 4. Red-Team Security Evaluation
- Total fixtures: `20`
- Baseline pass rate (unguarded B2): `0.4`
- Mitigated pass rate (guarded B6): `1.0`
- Category pass counts (mitigated): `{'policy_pii_leakage': '5/5', 'prompt_document_injection': '5/5', 'safety_robustness': '5/5', 'tool_misuse': '5/5'}`
