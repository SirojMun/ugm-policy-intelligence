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

        # Evidence-grounded explanatory extractive synthesis (preserves enumerations & deduplicates clauses)
        ql = query.lower()
        q_filler = {"apa", "saja", "yang", "dan", "atau", "dalam", "untuk", "bagaimana", "sebutkan", "jelaskan", "menurut", "ugm"}
        q_toks = set(t for t in tokenize_id(query) if t not in q_filler) or set(tokenize_id(query))
        is_list_query = any(k in ql for k in ("apa saja", "sebutkan", "daftar", "meliputi", "jenis", "macam"))
        mentions_pensiun = any(k in ql for k in ("pensiun", "purna tugas"))

        def _clean_trim(text: str, limit: int = 560) -> str:
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) <= limit:
                return text
            cut = text[:limit]
            # Prefer complete sentence boundary before limit, else word boundary
            last_dot = max(cut.rfind(". "), cut.rfind("; "))
            if last_dot >= int(limit * 0.45):
                return cut[: last_dot + 1].strip()
            return cut.rsplit(" ", 1)[0].rstrip(",;:") + "..."

        def _extract_clauses(raw_body: str) -> list[str]:
            lines = [ln.strip() for ln in raw_body.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            blocks: list[str] = []
            i = 0
            while i < len(lines):
                ln = lines[i]
                # Merge stem ending with ':' and its following enumerated items (a., 1., -)
                if ln.endswith(":"):
                    grp = [ln]
                    j = i + 1
                    while j < len(lines) and re.match(r"^(?:[a-z]\.|[0-9]+\.|[0-9]+\)|-)\s+", lines[j], re.IGNORECASE):
                        grp.append(lines[j])
                        j += 1
                    joined = " ".join(grp)
                    # Split only at sentence boundaries before numbered ayat "(2) ...", never mid-sentence after "ayat (1)"
                    ayat_parts = re.split(r"(?<=[.!?])\s+(?=\(\d+\)\s+)", joined)
                    blocks.extend(p.strip() for p in ayat_parts if len(p.strip()) >= 20)
                    i = j
                    continue
                if re.match(r"^\(\d+\)\s+", ln):
                    ayat_parts = re.split(r"(?<=[.!?])\s+(?=\(\d+\)\s+)", ln)
                    blocks.extend(p.strip() for p in ayat_parts if len(p.strip()) >= 20)
                else:
                    blocks.extend(seg.strip() for seg in re.split(r"(?<=[.!?])\s+", ln) if len(seg.strip()) >= 20)
                i += 1
            return [re.sub(r"\s+", " ", b).strip() for b in blocks if len(b.strip()) >= 20] or [raw_body[:320]]

        def _clause_score(clause: str, sc: ScoredChunk) -> float:
            c_toks = set(tokenize_id(clause))
            t_toks = set(tokenize_id(sc.chunk.title))
            score = len(q_toks & c_toks) / max(1, len(q_toks)) + 0.55 * (len(q_toks & t_toks) / max(1, len(q_toks)))
            if is_list_query and re.search(r"\b(?:meliputi|terdiri atas)\s*:", clause, re.IGNORECASE):
                score += 0.35
            if is_list_query and re.search(r"\bpasal\s+[123]\b", (sc.chunk.pasal or "").lower()) and re.search(r"\b(?:meliputi|terdiri atas)\s*:", clause, re.IGNORECASE):
                score += 0.25
            if not mentions_pensiun and "pensiunan" in clause.lower() and "dosen tetap" not in clause.lower().replace("pensiunan dosen tetap", ""):
                score -= 0.30
            if not mentions_pensiun and "pensiunan" in sc.chunk.section_path.lower():
                score -= 0.15
            return score

        candidates_scored: list[tuple[float, str, ScoredChunk]] = []
        for rank_idx, sc in enumerate(budget_chunks[:5]):
            raw_body = re.sub(r"^\[[^\]]+\]\s*", "", sc.chunk.text).strip()
            raw_body = re.sub(r"<!--.*?-->", "", raw_body, flags=re.DOTALL)
            raw_body = re.sub(r"\[SYSTEM OVERRIDE.*?\]", "", raw_body, flags=re.IGNORECASE)
            clauses = _extract_clauses(raw_body)
            best_c = max(clauses, key=lambda c: _clause_score(c, sc))
            total_s = _clause_score(best_c, sc) + max(0.0, 0.12 - 0.02 * rank_idx)
            candidates_scored.append((total_s, _clean_trim(best_c, 600), sc))

        candidates_scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate near-identical clauses so similar articles aren't repeated
        selected_sentences: list[tuple[str, ScoredChunk]] = []
        seen_tok_sets: list[set[str]] = []
        for _, cand_txt, sc in candidates_scored:
            c_set = set(tokenize_id(cand_txt))
            is_dup = any(
                (len(c_set & prev) / max(1, len(c_set | prev))) >= 0.72
                for prev in seen_tok_sets
            )
            if is_dup:
                continue
            seen_tok_sets.append(c_set)
            selected_sentences.append((cand_txt, sc))
            if len(selected_sentences) >= 3:
                break

        claims: list[str] = []
        prose_points: list[str] = []
        for sent, sc in selected_sentences:
            loc = sc.chunk.pasal or sc.chunk.section_path or "ketentuan terkait"
            ref = f"[{sc.chunk.doc_id}, {loc}, berlaku {sc.chunk.valid_time or sc.chunk.year or 'Unknown'}, status {sc.chunk.status}]"
            claims.append(f"{sent} {ref}")
            prose_points.append(f"• {sent} ({sc.chunk.doc_id}, {loc})")

        primary = selected_sentences[0][1].chunk if selected_sentences else budget_chunks[0].chunk
        header_note = (
            f"Berdasarkan {primary.title} "
            f"({primary.doc_id}, berlaku {primary.valid_time or primary.year or '-'}, status {primary.status}):"
        )
        answer_lines = [header_note, *prose_points]

        # Explicitly surface status transitions or Dicabut warnings
        statuses = {sc.chunk.status for sc in budget_chunks}
        if "Dicabut" in statuses:
            superseded = [sc.chunk for sc in budget_chunks if sc.chunk.status == "Dicabut"]
            active = [sc.chunk for sc in budget_chunks if sc.chunk.status != "Dicabut"]
            if superseded and active:
                note = f"Catatan keberlakuan: {superseded[0].doc_id} berstatus Dicabut; acuan mutakhir merujuk pada {active[0].doc_id} ({active[0].valid_time or active[0].year})."
                claims.insert(0, note)
                answer_lines.append(note)
            elif primary.status == "Dicabut":
                note = f"Catatan keberlakuan: {primary.doc_id} berstatus Dicabut dan dinyatakan tidak berlaku."
                claims.insert(0, note)
                answer_lines.append(note)

        answer = "\n".join(answer_lines)
        return {
            "answer": answer,
            "claims": claims,
            "verdict": "supported",
            "abstain_reason": "",
            "prompt_tokens_est": used_tokens,
            "completion_tokens_est": estimate_tokens(answer),
        }
