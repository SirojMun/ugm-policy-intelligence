#!/usr/bin/env python3
"""Kaggle GPU QLoRA Fine-Tuning Script for B5 (uses strictly train.jsonl + corpus_clean.jsonl)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunking import chunk_corpus, load_corpus
from src.retrieval import PolicyRetrievalIndex


def build_sft_dataset(
    train_jsonl: str = "data/benchmark/train.jsonl",
    corpus_jsonl: str = "corpus_clean.jsonl",
    out_jsonl: str = "data/sft/train_sft.jsonl",
) -> str:
    """Create strictly leak-free SFT examples from train.jsonl & retrieved corpus passages."""
    docs = load_corpus(corpus_jsonl)
    chunks = chunk_corpus(docs, "hierarchical")
    idx = PolicyRetrievalIndex(chunks)

    train_rows = [json.loads(l) for l in Path(train_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for row in train_rows:
            hits = idx.retrieve(row["query"], mode="hybrid_rerank", top_k=4)
            ctx = "\n---\n".join(
                f"[{h.chunk.doc_id} | {h.chunk.pasal or h.chunk.section_path} | valid_time={h.chunk.valid_time} | status={h.chunk.status}]\n{h.chunk.text[:800]}"
                for h in hits
            )
            target_json = json.dumps(
                {
                    "answer": row["gold_answer"] or "Bukti peraturan UGM tidak tersedia di dalam korpus untuk pertanyaan ini.",
                    "claims": [row["gold_answer"]] if row["gold_answer"] else [],
                    "verdict": row["gold_verdict"],
                    "valid_time": row.get("gold_valid_time", ""),
                    "abstain_reason": "" if row["gold_verdict"] == "supported" else "no_evidence_in_corpus",
                },
                ensure_ascii=False,
            )
            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Anda adalah asisten Academic Policy Intelligence UGM. Jawab hanya dari bukti terverifikasi dalam JSON.",
                    },
                    {"role": "user", "content": f"Pertanyaan: {row['query']}\n\nKonteks:\n{ctx}"},
                    {"role": "assistant", "content": target_json},
                ]
            }
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return str(out_path)


def train_qlora(
    sft_jsonl: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "models/b5_qlora_adapter",
    epochs: int = 3,
    lr: float = 2e-4,
) -> None:
    """Run 4-bit QLoRA training sized realistically for Kaggle T4/P100 (16GB VRAM)."""
    import inspect
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU tidak terdeteksi. Pelatihan QLoRA (B5) wajib dijalankan di GPU Kaggle (T4/P100 16GB) "
            "atau jalankan `--prepare-only` jika hanya ingin menyiapkan dataset SFT di CPU."
        )
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb < 6.0:
        raise RuntimeError(
            f"GPU terdeteksi hanya memiliki {vram_gb:.1f} GB VRAM ({torch.cuda.get_device_name(0)}), "
            "tidak cukup untuk pelatihan 4-bit QLoRA Qwen2.5-1.5B. Jalankan skrip ini di GPU Kaggle (16GB VRAM)."
        )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = {
        "quantization_config": bnb_cfg,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16, **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, **load_kw)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_cfg)
    # Force all non-4bit floating parameters & buffers to float32 and disable GradScaler (fp16=False, bf16=False)
    # so Turing/Pascal GPUs (Tesla T4/P100) never trigger AMP unscale BFloat16 crashes.
    for param in model.parameters():
        if type(param).__name__ != "Params4bit" and param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    for buf in model.buffers():
        if buf.is_floating_point() and buf.dtype != torch.float32:
            buf.data = buf.data.to(torch.float32)
    for mod in model.modules():
        qs = getattr(getattr(mod, "weight", None), "quant_state", None)
        if qs is not None and getattr(qs, "dtype", None) != torch.float32:
            qs.dtype = torch.float32

    trainable_dtypes = {str(p.dtype) for p in model.parameters() if p.requires_grad}
    print(f"QLoRA trainable param dtypes: {sorted(trainable_dtypes)}")

    dataset = load_dataset("json", data_files=sft_jsonl, split="train")
    sft_cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
    cfg_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": lr,
        "fp16": False,
        "bf16": False,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "report_to": "none",
        "seed": 42,
    }
    if "activation_checkpointing" in sft_cfg_params:
        cfg_kwargs["activation_checkpointing"] = True
    elif "gradient_checkpointing" in sft_cfg_params:
        cfg_kwargs["gradient_checkpointing"] = True
    if "max_seq_length" in sft_cfg_params:
        cfg_kwargs["max_seq_length"] = 1536
    elif "max_length" in sft_cfg_params:
        cfg_kwargs["max_length"] = 1536
    train_cfg = SFTConfig(**cfg_kwargs)

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs = {
        "model": model,
        "args": train_cfg,
        "train_dataset": dataset,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(output_dir)
    Path(output_dir, "training_metrics.json").write_text(
        json.dumps({"base_model": base_model, "metrics": result.metrics}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leak-free SFT data and run Kaggle QLoRA training for B5.")
    parser.add_argument("--prepare-only", action="store_true", help="Build train_sft.jsonl without starting GPU training.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="models/b5_qlora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    sft_path = build_sft_dataset()
    print(f"Built leak-free SFT train file: {sft_path}")
    if not args.prepare_only:
        train_qlora(sft_path, base_model=args.base_model, output_dir=args.output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
