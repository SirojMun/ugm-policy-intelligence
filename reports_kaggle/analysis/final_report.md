# Academic Policy Intelligence — Experiment & Audit Report (`reports_kaggle`)

## 1. Benchmark & Leakage Audit
- Splits: `{'train': 60, 'dev': 30, 'test': 110}`, Qrels: `239`
- Query-level dedup (`split_manifest.json`): `{'total_queries': 200, 'exact_duplicate_count': 0, 'cross_split_near_dup_count': 0, 'near_dup_threshold': 0.72, 'near_dups_sample': [], 'leakage_free': True}`
- Unique gold documents by split: `{'train': 47, 'dev': 27, 'test': 87}`, cross-split gold doc overlap: `{'train_test': 2, 'dev_test': 4, 'train_dev': 1}`

## 2. QLoRA Fine-Tuning Summary
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Metrics: `{'train_runtime': 993.1284, 'train_samples_per_second': 0.181, 'train_steps_per_second': 0.012, 'total_flos': 2160350519414784.0, 'train_loss': 1.3857335646947224, 'entropy': 1.36350736447743, 'mean_token_accuracy': 0.7462747352463859, 'num_tokens': 262191.0, 'epoch': 3.0}`

## 3. B0–B6 Comparison on Locked Test (Mean ± Std across Independent Runs)

| Config | Correctness | Verdict Acc | Evidence Recall | Citation Prec | Temporal Acc | Groundedness | False Confident | Mean Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.3777 ± 0.0079 | 0.6666 | 0.0788 | 0.0788 | 0.4152 | 0.0788 | 0.1485 | 5535.64 |
| B1 | 0.3429 ± 0.015 | 0.5848 | 0.0576 | 0.0576 | 0.4939 | 0.0576 | 0.1697 | 5478.65 |
| B2 | 0.54 ± 0.0191 | 0.5606 | 0.7121 | 0.5819 | 0.7364 | 0.5076 | 0.0879 | 8902.84 |
| B3 | 0.5305 ± 0.0118 | 0.5424 | 0.7485 | 0.6899 | 0.7666 | 0.5874 | 0.1303 | 8786.34 |
| B4 | 0.6832 ± 0.0046 | 0.8576 | 0.8091 | 0.7352 | 0.7727 | 0.9182 | 0.0666 | 7100.99 |
| B5 | 0.735 ± 0.0149 | 0.9121 | 0.8546 | 0.7782 | 0.8576 | 0.9621 | 0.0515 | 10751.11 |
| B6 | 0.7362 ± 0.0068 | 0.9273 | 0.8394 | 0.7653 | 0.8182 | 0.9879 | 0.0606 | 11828.2 |

## 4. Red-Team Security Evaluation
- Total fixtures: `20`
- Baseline pass rate (unguarded B2): `0.6`
- Mitigated pass rate (guarded B6): `0.95`
- Category pass counts (mitigated): `{'policy_pii_leakage': '5/5', 'prompt_document_injection': '5/5', 'safety_robustness': '4/5', 'tool_misuse': '5/5'}`
