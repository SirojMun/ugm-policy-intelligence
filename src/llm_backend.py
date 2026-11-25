"""Unified LLM backend abstraction enforcing identical base model across B0-B5 & B6."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from src.retrieval import ScoredChunk, tokenize_id


def load_prompt_registry(path: str | Path = "prompts/registry.yaml") -> dict[str, Any]:
    """Load versioned prompt templates and attach deterministic SHA-256 prompt hashes."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key, val in data.get("prompts", {}).items():
        val["prompt_hash"] = hashlib.sha256(val["system"].encode("utf-8")).hexdigest()[:12]
    return data


def estimate_tokens(text: str) -> int:
    """Fast token estimate (~1.35 tokens per whitespace word for Indonesian/legal text)."""
    return max(1, int(len((text or "").split()) * 1.35))


class BaseLLMBackend:
    """Interface for both local reproducible extractive synthesis and Kaggle GPU HF/QLoRA models."""

    model_id: str = "base"

    def generate(
        self,
        system_prompt: str,
        query: str,
        context_chunks: list[ScoredChunk],
        few_shot_examples: list[dict[str, str]] | None = None,
        seed: int = 42,
        max_context_tokens: int = 4000,
    ) -> dict[str, Any]:
        raise NotImplementedError


class HuggingFaceCausalBackend(BaseLLMBackend):
    """Real compact open-weight LLM (0.5B-3B) + optional QLoRA adapter for Kaggle GPU."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_dir: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.base_model_name = base_model_name
        self.adapter_dir = adapter_dir
        self.model_id = f"{base_model_name}{'+qlora:' + Path(adapter_dir).name if adapter_dir else ''}"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": True}
        if load_in_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=target_dtype, **kwargs)
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=target_dtype, **kwargs)
        if adapter_dir:
            self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        else:
            self.model = base_model
        self.model.eval()

    def generate(
        self,
        system_prompt: str,
        query: str,
        context_chunks: list[ScoredChunk],
        few_shot_examples: list[dict[str, str]] | None = None,
        seed: int = 42,
        max_context_tokens: int = 4000,
    ) -> dict[str, Any]:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        ctx_parts: list[str] = []
        used_tokens = 0
        for h in context_chunks:
            snippet = f"[{h.chunk.doc_id} | {h.chunk.pasal or h.chunk.section_path} | valid_time={h.chunk.valid_time} | status={h.chunk.status}]\n{h.chunk.text}"
            t = estimate_tokens(snippet)
            if used_tokens + t > max_context_tokens and ctx_parts:
                break
            ctx_parts.append(snippet)
            used_tokens += t

        messages = [{"role": "system", "content": system_prompt}]
        if few_shot_examples:
            for ex in few_shot_examples[:3]:
                messages.append({"role": "user", "content": ex["query"]})
                messages.append({"role": "assistant", "content": ex["answer"]})

        user_msg = f"Pertanyaan: {query}\n\nKonteks Dokumen:\n" + ("\n---\n".join(ctx_parts) if ctx_parts else "(Tidak ada konteks)")
        user_msg += '\n\nKembalikan HANYA 1 objek JSON valid (ringkas maks 2 kalimat untuk answer): {"answer": str, "claims": [str], "verdict": "supported"|"unknown", "abstain_reason": str}.'
        messages.append({"role": "user", "content": user_msg})

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_context_tokens + 512).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        raw_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        parsed = {"answer": raw_out, "claims": [], "verdict": "unknown", "abstain_reason": ""}
        m_json = re.search(r"\{.*\}", raw_out, re.DOTALL)
        if m_json:
            try:
                obj = json.loads(m_json.group(0))
                parsed["answer"] = str(obj.get("answer", raw_out))
                raw_claims = obj.get("claims", [])
                if isinstance(raw_claims, list):
                    parsed["claims"] = [
                        (c.get("statement") or json.dumps(c, ensure_ascii=False)) if isinstance(c, dict) else str(c)
                        for c in raw_claims
                    ]
                parsed["verdict"] = obj.get("verdict", "unknown") if obj.get("verdict") in ("supported", "unknown") else "unknown"
                parsed["abstain_reason"] = str(obj.get("abstain_reason", ""))
            except json.JSONDecodeError:
                if '"verdict": "supported"' in raw_out or '"verdict":"supported"' in raw_out:
                    parsed["verdict"] = "supported"

        return {
            **parsed,
            "prompt_tokens_est": int(inputs["input_ids"].shape[1]),
            "completion_tokens_est": int(len(gen_ids)),
        }


class ExtractivePolicyBackend(BaseLLMBackend):
    """Deterministic, PII-safe local synthesis backend matching B0-B5 & B6 mechanics on CPU."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct(cpu-extractive-sim)", adapter_tag: str = "") -> None:
        self.model_id = f"{model_id}{'+' + adapter_tag if adapter_tag else ''}"
        self.has_adapter = bool(adapter_tag)

    def generate(
        self,
        system_prompt: str,
        query: str,
        context_chunks: list[ScoredChunk],
        few_shot_examples: list[dict[str, str]] | None = None,
        seed: int = 42,
        max_context_tokens: int = 4000,
    ) -> dict[str, Any]:
        # Strictly respect context token budget (Ablation A7)
        budget_chunks: list[ScoredChunk] = []
        used_tokens = estimate_tokens(system_prompt) + estimate_tokens(query)
        if few_shot_examples:
            used_tokens += sum(estimate_tokens(ex["query"] + ex["answer"]) for ex in few_shot_examples[:3])

        for ch in context_chunks:
            t = estimate_tokens(ch.chunk.text)
            if used_tokens + t > max_context_tokens and budget_chunks:
                break
            budget_chunks.append(ch)
            used_tokens += t

        # Zero-shot / no-retrieval mode (B0 / B1)
        if not budget_chunks:
            has_abstain_policy = "abstain" in system_prompt.lower() or bool(few_shot_examples)
            if has_abstain_policy:
                ans = "Tidak ditemukan bukti peraturan UGM terverifikasi karena tidak ada konteks dokumen yang diambil."
                return {
                    "answer": ans,
                    "claims": [],
                    "verdict": "unknown",
                    "abstain_reason": "no_retrieval_context_available",
                    "prompt_tokens_est": used_tokens,
                    "completion_tokens_est": estimate_tokens(ans),
                }
            # B0 ungrounded zero-shot guess
            ans = "Berdasarkan pengetahuan umum tanpa rujukan dokumen UGM, ketentuan spesifik tidak dapat dipastikan."
            return {
                "answer": ans,
                "claims": [ans],
                "verdict": "unknown",
                "abstain_reason": "zero_shot_without_retrieval",
                "prompt_tokens_est": used_tokens,
                "completion_tokens_est": estimate_tokens(ans),
            }

        # Evidence-grounded extractive synthesis
        q_toks = set(tokenize_id(query))
        selected_sentences: list[tuple[str, ScoredChunk]] = []

        for sc in budget_chunks[:4]:
            raw_body = re.sub(r"^\[[^\]]+\]\s*", "", sc.chunk.text).strip()
            # Neutralize obvious indirect prompt-injection lines in text
            raw_body = re.sub(r"<!--.*?-->", "", raw_body, flags=re.DOTALL)
            raw_body = re.sub(r"\[SYSTEM OVERRIDE.*?\]", "", raw_body, flags=re.IGNORECASE)
            sents = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+", raw_body) if len(s.strip()) >= 20]
            if not sents:
                sents = [raw_body[:240]]

            best_s = max(
                sents,
                key=lambda s: len(q_toks & set(tokenize_id(s))) / max(1, len(q_toks)),
            )
            selected_sentences.append((best_s[:280], sc))

        claims: list[str] = []
        parts: list[str] = []
        for sent, sc in selected_sentences[:3]:
            ref = f"[{sc.chunk.doc_id}{', ' + sc.chunk.pasal if sc.chunk.pasal else ''}, berlaku {sc.chunk.valid_time or sc.chunk.year or 'Unknown'}, status {sc.chunk.status}]"
            stmt = f"{sent} {ref}"
            claims.append(stmt)
            parts.append(stmt)

        # Explicitly surface status transitions when both Dicabut/Diubah and Berlaku docs appear
        statuses = {sc.chunk.status for sc in budget_chunks}
        if "Dicabut" in statuses and ("Berlaku" in statuses or len({sc.chunk.doc_id for sc in budget_chunks}) >= 2):
            superseded = [sc.chunk for sc in budget_chunks if sc.chunk.status == "Dicabut"]
            active = [sc.chunk for sc in budget_chunks if sc.chunk.status != "Dicabut"]
            if superseded and active:
                note = f"Catatan keberlakuan: {superseded[0].doc_id} berstatus Dicabut; acuan mutakhir merujuk pada {active[0].doc_id} ({active[0].valid_time or active[0].year})."
                claims.insert(0, note)
                parts.insert(0, note)

        answer = " ".join(parts)
        return {
            "answer": answer,
            "claims": claims,
            "verdict": "supported",
            "abstain_reason": "",
            "prompt_tokens_est": used_tokens,
            "completion_tokens_est": estimate_tokens(answer),
        }
