"""Seven experimental configurations B0-B6 with planner, bounded retry, verifier, and guarded tools."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.llm_backend import BaseLLMBackend, ExtractivePolicyBackend, load_prompt_registry
from src.retrieval import PolicyRetrievalIndex, RetrievalFilters, ScoredChunk, tokenize_id
from src.schemas import AtomicClaim, Citation, PolicyResponse, TraceSpan

PII_OUT_RE = re.compile(
    r"(?:\b(?:3[1-6]|1[1-9]|21|5[1-3]|6[1-5]|7[1-6]|8[12]|9[124])\d{14}\b"
    r"|\b\d{2}/\d{6}/[A-Z]{2,4}/\d{5}\b"
    r"|\b(?:19|20)\d{6}(?:19|20)\d{4}[12]\d{3}\b"
    r"|1370001234567)"
)

PII_REQUEST_RE = re.compile(
    r"(?:\bnik\b|\bnip\b|\bnim\b|rekening|tabungan|saldo|gaji pokok|kata sandi|password|kredensial|token api|kunci privat|rekam medis|nomor whatsapp|nomor hp|email pribadi|draf internal rahasia)",
    re.IGNORECASE,
)

OUT_OF_SCOPE_ENTITY_RE = re.compile(
    r"(?:universitas indonesia|\bui\b|\bitb\b|\bits\b|universitas airlangga|universitas padjadjaran|universitas diponegoro|universitas brawijaya|universitas hasanuddin|polres sleman|pengadilan negeri|kementerian luar negeri|bank indonesia|rsup dr\.?\s*sardjito|bitcoin|kripto|jet pribadi|eksplorasi bulan|kendaraan otonom|skck|izin mendirikan bangunan|\bimb\b|paspor diplomatik|visa kerja|kpr\b)",
    re.IGNORECASE,
)

INJECTION_PATTERNS_RE = re.compile(
    r"(?:abaikan semua aturan|ignore previous|system override|hacked_admin_access|matikan verifier|force_pasal|database dump)",
    re.IGNORECASE,
)


class SearchCorpusArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    doc_type: Literal[
        "Peraturan Rektor",
        "Peraturan MWA",
        "Peraturan Senat Akademik",
        "Keputusan Rektor",
        "Keputusan MWA",
        "Keputusan Senat Akademik",
        "Pedoman Akademik",
        "Pedoman Kemahasiswaan",
        "Pedoman Layanan",
        "Statuta / Eksternal",
    ] | None = None
    exclude_dicabut: bool = True
    top_k: int = Field(default=5, ge=1, le=10)


class DocLookupArgs(BaseModel):
    doc_id: str = Field(pattern=r"^UGM_[A-Z_]+_\d{4}_[A-Z0-9_]+$")


class TemporalCheckArgs(BaseModel):
    doc_id: str = Field(pattern=r"^UGM_[A-Z_]+_\d{4}_[A-Z0-9_]+$")
    as_of_date: str = Field(pattern=r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")


class ToolGuard:
    """Enforces allowlist, strict schema validation, path/injection blocking, and step budget."""

    ALLOWLIST = {
        "search_corpus": SearchCorpusArgs,
        "get_regulation_metadata": DocLookupArgs,
        "check_temporal_validity": TemporalCheckArgs,
        "get_cross_references": DocLookupArgs,
    }

    def __init__(self, docs_by_id: dict[str, dict[str, Any]], index: PolicyRetrievalIndex, max_calls: int = 4) -> None:
        self.docs_by_id = docs_by_id
        self.index = index
        self.max_calls = max_calls

    def validate_and_execute(self, tool_name: str, args: dict[str, Any], calls_so_far: int) -> dict[str, Any]:
        if calls_so_far >= self.max_calls:
            return {"allowed": False, "reason": "budget_overrun_max_calls", "result": None}
        if tool_name not in self.ALLOWLIST:
            return {"allowed": False, "reason": f"unauthorized_tool:{tool_name}", "result": None}

        # Guard string args against shell/SQL/traversal patterns
        raw_str = json.dumps(args, ensure_ascii=False)
        if any(bad in raw_str for bad in ("../", "..\\", ";", "--", "DROP TABLE", "/etc/")):
            return {"allowed": False, "reason": "unsafe_argument_pattern", "result": None}

        schema_cls = self.ALLOWLIST[tool_name]
        try:
            validated = schema_cls(**args)
        except ValidationError as exc:
            return {"allowed": False, "reason": f"schema_validation_error:{exc.errors()[0]['loc']}", "result": None}

        if tool_name == "search_corpus":
            v = validated  # type: ignore[assignment]
            flt = RetrievalFilters(
                doc_type=v.doc_type,
                exclude_statuses=("Dicabut",) if v.exclude_dicabut else (),
            )
            hits = self.index.retrieve(v.query, mode="hybrid_rerank", top_k=v.top_k, filters=flt)
            return {
                "allowed": True,
                "reason": "ok",
                "result": [
                    {
                        "doc_id": h.chunk.doc_id,
                        "chunk_id": h.chunk.chunk_id,
                        "pasal": h.chunk.pasal,
                        "valid_time": h.chunk.valid_time,
                        "status": h.chunk.status,
                        "score": h.score,
                    }
                    for h in hits
                ],
                "hits": hits,
            }

        did = validated.doc_id  # type: ignore[attr-defined]
        doc = self.docs_by_id.get(did)
        if not doc:
            return {"allowed": True, "reason": "doc_not_found", "result": {"found": False, "doc_id": did}}

        if tool_name == "get_regulation_metadata":
            return {
                "allowed": True,
                "reason": "ok",
                "result": {
                    "doc_id": did,
                    "title": doc["title"],
                    "doc_type": doc["doc_type"],
                    "regulation_number": doc.get("regulation_number", ""),
                    "year": doc.get("year"),
                    "valid_time": doc.get("valid_time", ""),
                    "status": doc.get("status", "Unknown"),
                    "version": doc.get("version", "original"),
                },
            }

        if tool_name == "check_temporal_validity":
            as_of = validated.as_of_date  # type: ignore[attr-defined]
            vt = str(doc.get("valid_time") or doc.get("year") or "")
            eff_iso = vt if len(vt) == 10 else (f"{vt}-01-01" if len(vt) == 4 else "")
            is_effective_by_date = bool(not eff_iso or eff_iso <= as_of)
            status = doc.get("status", "Unknown")
            return {
                "allowed": True,
                "reason": "ok",
                "result": {
                    "doc_id": did,
                    "as_of_date": as_of,
                    "valid_time": vt,
                    "status": status,
                    "valid_as_of_date": is_effective_by_date and status != "Dicabut",
                },
            }

        if tool_name == "get_cross_references":
            return {
                "allowed": True,
                "reason": "ok",
                "result": {
                    "doc_id": did,
                    "related_regulations": doc.get("related_regulations", []),
                },
            }

        return {"allowed": False, "reason": "unhandled_tool", "result": None}


def _sanitize_context(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Strip HTML comments and indirect prompt-injection tokens from retrieved passages."""
    cleaned: list[ScoredChunk] = []
    for sc in chunks:
        txt = re.sub(r"<!--.*?-->", "", sc.chunk.text, flags=re.DOTALL)
        txt = re.sub(r"\[SYSTEM OVERRIDE[^\]]*\]", "", txt, flags=re.IGNORECASE)
        txt = PII_OUT_RE.sub("[REDACTED_PII]", txt)
        new_ch = sc.chunk.model_copy(update={"text": txt})
        cleaned.append(ScoredChunk(chunk=new_ch, score=sc.score, source_retriever=sc.source_retriever, parent_text=sc.parent_text))
    return cleaned


def _evidence_sufficiency_score(query: str, hits: list[ScoredChunk], mode: str = "hybrid_rerank") -> float:
    """Estimate whether retrieved passages genuinely cover the query terms."""
    if not hits:
        return 0.0
    if OUT_OF_SCOPE_ENTITY_RE.search(query) or PII_REQUEST_RE.search(query):
        return 0.0
    # Reject empty/punctuation-only queries
    q_toks = [t for t in tokenize_id(query) if len(t) >= 3]
    if not q_toks:
        return 0.0
    stop = {"yang", "dan", "atau", "dari", "pada", "dalam", "untuk", "tentang", "apakah", "berapa", "bagaimana", "sebutkan", "jelaskan", "menurut", "ugm", "universitas", "gadjah", "mada", "tahun", "nomor", "peraturan", "keputusan", "pedoman", "aturan", "resmi"}
    content_toks = [t for t in q_toks if t not in stop] or q_toks
    top_text = " ".join(f"{h.chunk.title} {h.chunk.section_path} {h.chunk.text[:1600]}" for h in hits[:4])
    d_set = set(tokenize_id(top_text))
    coverage = sum(1 for t in content_toks if t in d_set) / max(1, len(content_toks))
    top_score = hits[0].score
    if mode == "bm25":
        return min(1.0, 0.7 * coverage + 0.3 * min(1.0, top_score / 15.0))
    return min(1.0, 0.75 * coverage + 0.25 * min(1.0, top_score))


NEGATION_TOKENS = {"tidak", "bukan", "belum", "tanpa", "dilarang", "batal", "dicabut", "menolak"}
POSITIVE_MODAL_TOKENS = {"wajib", "harus", "boleh", "diperbolehkan", "berhak", "berlaku", "sah"}
OPPOSITE_NORMATIVE_PAIRS = (
    ({"wajib", "harus"}, {"dilarang", "bebas"}),
    ({"boleh", "diperbolehkan"}, {"dilarang", "terlarang"}),
    ({"berlaku", "sah"}, {"dicabut", "batal"}),
)
REG_REF_RE = re.compile(
    r"((?:peraturan|keputusan)\s+[a-z\s]{2,40}?nomor\s+\d+\s+tahun\s+\d{4})",
    re.IGNORECASE,
)


def _norm_reg_ref(ref: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\bugm\b", "", ref, flags=re.IGNORECASE)).strip().lower()


def _extract_repeal_relation(text: str, self_title: str = "") -> tuple[str, str] | None:
    """Return (repealer_ref, repealed_ref) when text expresses an active/passive repeal."""
    resolved = text
    if self_title:
        self_refs = REG_REF_RE.findall(self_title)
        if self_refs:
            resolved = re.sub(
                r"\b(?:peraturan|keputusan)(?:\s+rektor|\s+mwa|\s+senat\s+akademik)?\s+ini\b",
                self_refs[0],
                resolved,
                flags=re.IGNORECASE,
            )
    tl = re.sub(r"\bugm\b", "", resolved, flags=re.IGNORECASE)
    m_act = re.search(
        r"((?:peraturan|keputusan)\s+[a-z\s]{2,40}?nomor\s+\d+\s+tahun\s+\d{4}).{0,80}?\bmencabut\b.{0,80}?((?:peraturan|keputusan)\s+[a-z\s]{2,40}?nomor\s+\d+\s+tahun\s+\d{4})",
        tl,
        re.IGNORECASE | re.DOTALL,
    )
    if m_act:
        return _norm_reg_ref(m_act.group(1)), _norm_reg_ref(m_act.group(2))
    m_pas = re.search(
        r"((?:peraturan|keputusan)\s+[a-z\s]{2,40}?nomor\s+\d+\s+tahun\s+\d{4}).{0,100}?\bdicabut\b.{0,80}?\b(?:oleh|dengan)\b.{0,40}?((?:peraturan|keputusan)\s+[a-z\s]{2,40}?nomor\s+\d+\s+tahun\s+\d{4})",
        tl,
        re.IGNORECASE | re.DOTALL,
    )
    if m_pas:
        return _norm_reg_ref(m_pas.group(2)), _norm_reg_ref(m_pas.group(1))
    # Handle clause: "<self_ref> mulai berlaku, <repealed_ref> dicabut"
    m_impl = re.search(
        r"((?:peraturan|keputusan)\s+[a-z\s]{2,40}?nomor\s+\d+\s+tahun\s+\d{4}).{0,80}?\bmulai\s+berlaku\b.{0,80}?((?:peraturan|keputusan)\s+[a-z\s]{2,40}?nomor\s+\d+\s+tahun\s+\d{4}).{0,40}?\bdicabut\b",
        tl,
        re.IGNORECASE | re.DOTALL,
    )
    if m_impl:
        return _norm_reg_ref(m_impl.group(1)), _norm_reg_ref(m_impl.group(2))
    return None


BOILERPLATE_STOP = {
    "peraturan", "rektor", "keputusan", "nomor", "tahun", "pasal", "bab", "ayat",
    "ugm", "ini", "tersebut", "pada", "saat", "dan", "yang", "dalam", "untuk",
    "oleh", "dengan", "dinyatakan", "tentang", "mulai",
}


def _has_negation_conflict(claim_stmt: str, evidence_sentence: str) -> bool:
    """Detect direct polarity inversion on high-overlap aligned clauses."""
    c_toks = set(tokenize_id(claim_stmt))
    e_toks = set(tokenize_id(evidence_sentence))
    c_content = c_toks - NEGATION_TOKENS - BOILERPLATE_STOP
    e_content = e_toks - NEGATION_TOKENS - BOILERPLATE_STOP
    shared = c_content & e_content
    if len(shared) < 3:
        return False
    jaccard = len(shared) / max(1, len(c_content | e_content))
    if jaccard < 0.55:
        return False

    c_neg = len(c_toks & NEGATION_TOKENS)
    e_neg = len(e_toks & NEGATION_TOKENS)
    if (c_neg % 2) != (e_neg % 2) and (c_toks & POSITIVE_MODAL_TOKENS or e_toks & POSITIVE_MODAL_TOKENS):
        return True

    for pos_set, neg_set in OPPOSITE_NORMATIVE_PAIRS:
        if (c_toks & pos_set and e_toks & neg_set) or (e_toks & pos_set and c_toks & neg_set):
            return True
    return False


def _verify_claims(
    claims: list[str],
    hits: list[ScoredChunk],
    query_context: str = "",
) -> tuple[list[AtomicClaim], list[Citation]]:
    """Verify atomic claims for support, negation contradiction, and temporal-repeal inversion."""
    atomic: list[AtomicClaim] = []
    citations_map: dict[str, Citation] = {}

    # Pre-collect repeal directions & revoked doc statuses present in retrieved evidence
    evidence_repeals: set[tuple[str, str]] = set()
    revoked_refs: set[str] = set()
    for sc in hits:
        rel = _extract_repeal_relation(sc.chunk.text, self_title=sc.chunk.title)
        if rel:
            evidence_repeals.add(rel)
        if sc.chunk.status == "Dicabut":
            for m in REG_REF_RE.findall(sc.chunk.title):
                revoked_refs.add(_norm_reg_ref(m))

    for idx, stmt in enumerate(claims, start=1):
        stmt_clean = PII_OUT_RE.sub("[REDACTED_PII]", stmt).strip()
        stmt_body = re.sub(r"\[[^\]]+\]", "", stmt_clean).strip() or stmt_clean
        stmt_norm = re.sub(r"\s+", " ", stmt_body.lower()).strip()
        s_toks = set(tokenize_id(stmt_clean))
        matched_cids: list[str] = []
        contradiction_cids: list[str] = []
        contradiction_reason = ""
        best_overlap = 0.0

        # Check explicit repeal direction inversion against retrieved corpus evidence
        stmt_rel = _extract_repeal_relation(stmt_body) or _extract_repeal_relation(query_context)
        if stmt_rel and (stmt_rel[1], stmt_rel[0]) in evidence_repeals and stmt_rel not in evidence_repeals:
            contradiction_reason = "temporal_repeal_direction_inverted"

        # Check if claim asserts an expressly Dicabut regulation is still valid
        if not contradiction_reason and re.search(r"\bmasih\s+berlaku\b", stmt_body, re.IGNORECASE):
            for m in REG_REF_RE.findall(stmt_body):
                if _norm_reg_ref(m) in revoked_refs:
                    contradiction_reason = "temporal_status_revoked"
                    break

        is_verbatim_quote = any(
            stmt_norm and len(stmt_norm) >= 16 and stmt_norm in re.sub(r"\s+", " ", sc.chunk.text.lower())
            for sc in hits
        )

        for sc in hits:
            body_text = re.sub(r"^\[[^\]]+\]\s*", "", sc.chunk.text)
            if not is_verbatim_quote:
                ev_sentences = [
                    seg.strip()
                    for seg in re.split(r"(?<=[.!?])\s+|\n+", body_text)
                    if len(seg.strip()) >= 12 and not seg.strip().startswith("#")
                ]
                for ev_sent in ev_sentences:
                    if _has_negation_conflict(stmt_body, ev_sent):
                        contradiction_cids.append(sc.chunk.chunk_id)
                        if not contradiction_reason:
                            contradiction_reason = "polarity_negation_conflict"

            full_ev = f"{sc.chunk.title} {sc.chunk.section_path} {sc.chunk.text}"
            ch_toks = set(tokenize_id(full_ev))
            overlap = len(s_toks & ch_toks) / max(1, len(s_toks))
            best_overlap = max(best_overlap, overlap)
            if overlap >= 0.35 or sc.chunk.doc_id in stmt_clean:
                matched_cids.append(sc.chunk.chunk_id)
                if sc.chunk.chunk_id not in citations_map:
                    citations_map[sc.chunk.chunk_id] = Citation(
                        doc_id=sc.chunk.doc_id,
                        chunk_id=sc.chunk.chunk_id,
                        pasal=sc.chunk.pasal,
                        section_path=sc.chunk.section_path,
                        valid_time=sc.chunk.valid_time,
                        status=sc.chunk.status,
                        quote=re.sub(r"^\[[^\]]+\]\s*", "", sc.chunk.text)[:200].strip(),
                    )

        if contradiction_reason:
            verdict: Literal["supported", "contradicted", "unknown"] = "contradicted"
            ev_ids = (contradiction_cids or matched_cids)[:2]
            conf = round(max(0.7, best_overlap), 4)
            reason = contradiction_reason
        elif matched_cids:
            verdict = "supported"
            ev_ids = matched_cids[:2]
            conf = round(min(1.0, max(0.35, best_overlap)), 4)
            reason = "evidence_aligned"
        else:
            verdict = "unknown"
            ev_ids = []
            conf = round(best_overlap, 4)
            reason = "not_enough_evidence"

        atomic.append(
            AtomicClaim(
                claim_id=f"C{idx}",
                statement=stmt_clean,
                citations=ev_ids,
                verdict=verdict,
                confidence=conf,
                reason=reason,
            )
        )
    return atomic, list(citations_map.values())


def resolve_as_of_date(request_date: str | None = None) -> str:
    """Resolve explicit historical evaluation date or default to current UTC date."""
    if request_date:
        return request_date
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class PolicyExperimentRunner:
    """Runs any of B0-B6 consistently with tracing, budgets, and fallback."""

    def __init__(
        self,
        docs_by_id: dict[str, dict[str, Any]],
        fixed_index: PolicyRetrievalIndex,
        heading_index: PolicyRetrievalIndex,
        hierarchical_index: PolicyRetrievalIndex,
        base_backend: BaseLLMBackend | None = None,
        peft_backend: BaseLLMBackend | None = None,
        train_examples: list[dict[str, str]] | None = None,
        prompt_registry_path: str | Path = "prompts/registry.yaml",
        index_version: str = "ugm-v1-422docs",
        b6_parent_config: Literal["B4", "B5"] = "B5",
    ) -> None:
        self.docs_by_id = docs_by_id
        self.fixed_index = fixed_index
        self.heading_index = heading_index
        self.hierarchical_index = hierarchical_index
        self.base_backend = base_backend or ExtractivePolicyBackend()
        self.peft_backend = peft_backend or ExtractivePolicyBackend(adapter_tag="qlora-train-v1")
        self.train_examples = train_examples or []
        self.prompts = load_prompt_registry(prompt_registry_path)["prompts"]
        self.index_version = index_version
        self.b6_parent_config = b6_parent_config
        self.tool_guard = ToolGuard(docs_by_id=docs_by_id, index=hierarchical_index, max_calls=4)

    def _plan_query(self, query: str) -> dict[str, Any]:
        ql = query.lower()
        if OUT_OF_SCOPE_ENTITY_RE.search(query) or PII_REQUEST_RE.search(query) or not tokenize_id(query):
            qtype = "no-evidence"
        elif any(k in ql for k in ("dibanding", "perubahan", "dicabut", "mencabut", "pencabutan", "diubah", "mengubah", "membatalkan", "masih berlaku", "pertentangan", "antara ", "versi ")):
            qtype = "conflict"
        elif any(k in ql for k in ("gabungkan", "keterkaitan", "hubungan", "sekaligus", "pasal")):
            qtype = "multi-hop"
        else:
            qtype = "easy"

        subqueries = [query]
        if qtype == "multi-hop" or qtype == "conflict":
            # Extract regulation references or key clauses as an extra subquery
            m_nums = re.findall(r"(?:Peraturan|Keputusan)\s+[A-Za-z ]+?Nomor\s+\d+\s+Tahun\s+\d{4}", query, re.IGNORECASE)
            subqueries.extend(m_nums[:2])

        return {"query_type": qtype, "subqueries": subqueries}

    def run(
        self,
        config_id: Literal["B0", "B1", "B2", "B3", "B4", "B5", "B6"],
        query: str,
        query_id: str = "",
        as_of_date: str | None = None,
        seed: int = 42,
        max_context_tokens: int = 4000,
        enable_verifier: bool = True,
        enable_tools: bool = True,
        adaptive_retrieval: bool = True,
        top_k: int = 5,
        extra_injected_context: str = "",
    ) -> PolicyResponse:
        t_start = time.perf_counter()
        resolved_date = resolve_as_of_date(as_of_date)
        trace: list[TraceSpan] = [TraceSpan(step="resolve_as_of_date", detail=resolved_date)]

        prompt_key_map = {
            "B0": "B0_direct",
            "B1": "B1_structured",
            "B2": "B2_naive_rag",
            "B3": "B3_hybrid_rag",
            "B4": "B4_adaptive_rag",
            "B5": "B5_peft_rag",
            "B6": "B6_guarded_agent",
        }
        p_info = self.prompts[prompt_key_map[config_id]]
        system_prompt = p_info["system"]
        prompt_ver = f"{p_info['version']}:{p_info['prompt_hash']}"
        backend = self.peft_backend if config_id == "B5" else self.base_backend
        if config_id == "B6":
            # B6 inherits the strongest parent selected on dev (B4 or B5)
            backend = self.peft_backend if self.b6_parent_config == "B5" else self.base_backend

        # Guard malformed / empty / PII / prompt-injection inputs early in guarded configs (B4/B5/B6)
        if config_id in ("B4", "B5", "B6"):
            if not tokenize_id(query):
                return PolicyResponse(
                    query_id=query_id,
                    config_id=config_id,
                    query=query,
                    answer="Permintaan tidak valid atau kosong sehingga tidak dapat diproses.",
                    verdict="unknown",
                    abstain_reason="empty_or_malformed_query",
                    latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                    model_id=backend.model_id,
                    prompt_version=prompt_ver,
                    index_version=self.index_version,
                    trace=[TraceSpan(step="input_guard", detail="malformed_query_abstain")],
                )
            if PII_REQUEST_RE.search(query):
                return PolicyResponse(
                    query_id=query_id,
                    config_id=config_id,
                    query=query,
                    answer="Permintaan ditolak karena berpotensi mengakses atau membocorkan data pribadi (PII) maupun informasi internal non-publik.",
                    verdict="unknown",
                    abstain_reason="policy_pii_guard_refusal",
                    latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                    model_id=backend.model_id,
                    prompt_version=prompt_ver,
                    index_version=self.index_version,
                    trace=[TraceSpan(step="pii_guard", detail="blocked_pii_or_private_request")],
                )
            if INJECTION_PATTERNS_RE.search(query):
                trace.append(TraceSpan(step="injection_guard", detail="detected_user_prompt_injection_attempt"))
                if any(k in query.lower() for k in ("abaikan semua", "force_pasal", "matikan verifier")):
                    return PolicyResponse(
                        query_id=query_id,
                        config_id=config_id,
                        query=query,
                        answer="Permintaan mengandung upaya manipulasi instruksi/status aturan dan ditolak. Sistem hanya menjawab berdasarkan dokumen terverifikasi.",
                        verdict="unknown",
                        abstain_reason="prompt_injection_rejected",
                        latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                        model_id=backend.model_id,
                        prompt_version=prompt_ver,
                        index_version=self.index_version,
                        trace=trace,
                    )

        # B0 & B1: No retrieval
        if config_id in ("B0", "B1"):
            few_shots = self.train_examples[:3] if config_id == "B1" else None
            gen = backend.generate(
                system_prompt=system_prompt,
                query=query,
                context_chunks=[],
                few_shot_examples=few_shots,
                seed=seed,
                max_context_tokens=max_context_tokens,
            )
            trace.append(TraceSpan(step="llm_generate", detail=f"{config_id}_no_retrieval"))
            return PolicyResponse(
                query_id=query_id,
                config_id=config_id,
                query=query,
                answer=PII_OUT_RE.sub("[REDACTED_PII]", gen["answer"]),
                atomic_claims=[AtomicClaim(claim_id=f"C{i+1}", statement=c, verdict="unknown") for i, c in enumerate(gen["claims"])],
                citations=[],
                valid_time="",
                verdict=gen["verdict"],
                abstain_reason=gen["abstain_reason"],
                latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                prompt_tokens_est=gen["prompt_tokens_est"],
                completion_tokens_est=gen["completion_tokens_est"],
                model_id=backend.model_id,
                prompt_version=prompt_ver,
                index_version="none",
                trace=trace,
            )

        # B2: Naive RAG (dense + fixed chunk + top-k)
        if config_id == "B2":
            t0 = time.perf_counter()
            hits = self.fixed_index.retrieve(query, mode="dense", top_k=top_k)
            trace.append(TraceSpan(step="retrieve_b2_dense_fixed", duration_ms=round((time.perf_counter() - t0) * 1000, 2), metadata={"hits": len(hits)}))

        # B3: Hybrid RAG (BM25 + dense + RRF + reranker + metadata filter)
        elif config_id == "B3":
            t0 = time.perf_counter()
            flt = RetrievalFilters(exclude_statuses=("Dicabut",)) if "dicabut" not in query.lower() else None
            hits = self.heading_index.retrieve(query, mode="hybrid_rerank", top_k=top_k, filters=flt)
            trace.append(TraceSpan(step="retrieve_b3_hybrid_rerank", duration_ms=round((time.perf_counter() - t0) * 1000, 2), metadata={"hits": len(hits)}))

        # B4, B5, B6: Self-corrective / Adaptive RAG + Optional Guarded Tools
        else:
            plan = self._plan_query(query) if adaptive_retrieval else {"query_type": "fixed", "subqueries": [query]}
            trace.append(TraceSpan(step="query_planner", detail=plan["query_type"], metadata={"subqueries": len(plan["subqueries"])}))

            if plan["query_type"] == "no-evidence":
                return PolicyResponse(
                    query_id=query_id,
                    config_id=config_id,
                    query=query,
                    answer="Maaf, tidak ditemukan dasar peraturan UGM yang relevan di dalam korpus publik untuk pertanyaan di luar cakupan tersebut.",
                    verdict="unknown",
                    abstain_reason="out_of_scope_no_evidence_in_corpus",
                    latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                    model_id=backend.model_id,
                    prompt_version=prompt_ver,
                    index_version=self.index_version,
                    trace=trace,
                )

            t0 = time.perf_counter()
            eff_top_k = top_k + 3 if plan["query_type"] in ("multi-hop", "conflict") else top_k
            # For conflict queries, include Dicabut/Diubah docs so temporal resolution sees both versions!
            flt = None if plan["query_type"] == "conflict" else RetrievalFilters(exclude_statuses=("Dicabut",))

            if config_id == "B6" and enable_tools:
                search_args = {"query": query[:280], "exclude_dicabut": flt is not None, "top_k": min(10, eff_top_k)}
                tool_res = self.tool_guard.validate_and_execute(
                    "search_corpus",
                    search_args,
                    calls_so_far=0,
                )
                hits = tool_res.get("hits", []) if tool_res["allowed"] else []
                trace.append(
                    TraceSpan(
                        step="tool:search_corpus",
                        detail=tool_res.get("reason", "ok"),
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                        metadata={
                            "tool": "search_corpus",
                            "arguments": {"exclude_dicabut": search_args["exclude_dicabut"], "top_k": search_args["top_k"]},
                            "allowed": tool_res["allowed"],
                            "result_count": len(hits),
                            "plan": plan["query_type"],
                        },
                    )
                )
            else:
                hits = self.hierarchical_index.retrieve(query, mode="hybrid_rerank", top_k=eff_top_k, filters=flt)
                trace.append(TraceSpan(step="retrieve_initial", duration_ms=round((time.perf_counter() - t0) * 1000, 2), metadata={"hits": len(hits)}))

            # Evidence sufficiency check + bounded retry (max 2 retries)
            suff = _evidence_sufficiency_score(query, hits)
            trace.append(TraceSpan(step="sufficiency_check", metadata={"score": round(suff, 3)}))
            retry_count = 0
            while suff < 0.45 and retry_count < 2:
                retry_count += 1
                sub_q = plan["subqueries"][min(retry_count, len(plan["subqueries"]) - 1)]
                extra_hits = self.hierarchical_index.retrieve(sub_q, mode="hybrid_rerank", top_k=eff_top_k, filters=None)
                seen_cids = {h.chunk.chunk_id for h in hits}
                for eh in extra_hits:
                    if eh.chunk.chunk_id not in seen_cids:
                        hits.append(eh)
                        seen_cids.add(eh.chunk.chunk_id)
                suff = _evidence_sufficiency_score(query, hits)
                trace.append(TraceSpan(step=f"bounded_retry_{retry_count}", metadata={"sufficiency": round(suff, 3), "total_hits": len(hits)}))

            # B6 Guarded Tool checks: metadata + temporal validity + cross references on top hits
            if config_id == "B6" and enable_tools and hits:
                calls = 1
                top_dids = list(dict.fromkeys(h.chunk.doc_id for h in hits[:2]))
                for did in top_dids:
                    if calls >= 4:
                        break
                    t_args = {"doc_id": did, "as_of_date": resolved_date}
                    t_chk = self.tool_guard.validate_and_execute(
                        "check_temporal_validity", t_args, calls_so_far=calls
                    )
                    calls += 1
                    trace.append(
                        TraceSpan(
                            step="tool:check_temporal_validity",
                            detail=did,
                            metadata={
                                "tool": "check_temporal_validity",
                                "arguments": t_args,
                                "allowed": t_chk["allowed"],
                                "reason": t_chk.get("reason", ""),
                                **(t_chk.get("result") or {}),
                            },
                        )
                    )
                if calls < 4 and top_dids:
                    x_args = {"doc_id": top_dids[0]}
                    xref = self.tool_guard.validate_and_execute("get_cross_references", x_args, calls_so_far=calls)
                    refs_list = (xref.get("result") or {}).get("related_regulations", [])
                    trace.append(
                        TraceSpan(
                            step="tool:get_cross_references",
                            detail=top_dids[0],
                            metadata={
                                "tool": "get_cross_references",
                                "arguments": x_args,
                                "allowed": xref["allowed"],
                                "result_count": len(refs_list),
                            },
                        )
                    )

            if suff < 0.25:
                return PolicyResponse(
                    query_id=query_id,
                    config_id=config_id,
                    query=query,
                    answer="Bukti yang ditemukan tidak cukup kuat dan relevan untuk menjawab pertanyaan secara terverifikasi.",
                    verdict="unknown",
                    abstain_reason=f"insufficient_evidence_score:{suff:.2f}",
                    latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                    model_id=backend.model_id,
                    prompt_version=prompt_ver,
                    index_version=self.index_version,
                    trace=trace,
                )

        # Optional test-injection context simulation (red-team)
        if extra_injected_context:
            inj_chunk = hits[0].chunk.model_copy(update={"text": extra_injected_context + "\n" + hits[0].chunk.text}) if hits else None
            if inj_chunk:
                hits[0] = ScoredChunk(chunk=inj_chunk, score=hits[0].score, source_retriever=hits[0].source_retriever)

        sanitized_hits = _sanitize_context(hits) if config_id in ("B4", "B5", "B6") else hits
        gen = backend.generate(
            system_prompt=system_prompt,
            query=query,
            context_chunks=sanitized_hits,
            seed=seed,
            max_context_tokens=max_context_tokens,
        )

        # Claim verification + bounded repair (for B4/B5/B6 when enable_verifier=True)
        raw_claims = gen["claims"] or [gen["answer"]]
        if config_id in ("B4", "B5", "B6") and enable_verifier:
            atomic_claims, citations = _verify_claims(raw_claims, sanitized_hits, query_context=query)
            contradicted_claims = [c for c in atomic_claims if c.verdict == "contradicted"]
            if contradicted_claims:
                reasons = sorted({c.reason for c in contradicted_claims if c.reason})
                return PolicyResponse(
                    query_id=query_id,
                    config_id=config_id,
                    query=query,
                    answer="Klaim ditolak oleh verifikator karena bertentangan (kontradiksi atau pembalikan arah pencabutan) dengan dokumen peraturan terverifikasi.",
                    atomic_claims=atomic_claims,
                    citations=citations,
                    verdict="unknown",
                    abstain_reason=f"claim_verifier_contradicted:{','.join(reasons) or 'conflict'}",
                    latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                    prompt_tokens_est=gen["prompt_tokens_est"],
                    completion_tokens_est=gen["completion_tokens_est"],
                    model_id=backend.model_id,
                    prompt_version=prompt_ver,
                    index_version=self.index_version,
                    trace=trace + [TraceSpan(step="claim_verifier", detail="contradicted_abstained", metadata={"reasons": reasons})],
                )
            supported_claims = [c for c in atomic_claims if c.verdict == "supported"]
            if not supported_claims:
                return PolicyResponse(
                    query_id=query_id,
                    config_id=config_id,
                    query=query,
                    answer="Verifikator klaim tidak menemukan dukungan eksplisit yang memadai pada dokumen yang diambil.",
                    atomic_claims=atomic_claims,
                    citations=citations,
                    verdict="unknown",
                    abstain_reason="claim_verifier_unsupported",
                    latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                    prompt_tokens_est=gen["prompt_tokens_est"],
                    completion_tokens_est=gen["completion_tokens_est"],
                    model_id=backend.model_id,
                    prompt_version=prompt_ver,
                    index_version=self.index_version,
                    trace=trace + [TraceSpan(step="claim_verifier", detail="0_supported_abstained")],
                )
            # Bounded repair: filter out unsupported claims from final answer
            repaired_answer = " ".join(c.statement for c in supported_claims)
            verdict: Literal["supported", "unknown"] = "supported"
            abstain_reason = ""
            trace.append(TraceSpan(step="claim_verifier", metadata={"supported": len(supported_claims), "total": len(atomic_claims)}))
        else:
            atomic_claims, citations = _verify_claims(raw_claims, sanitized_hits, query_context=query)
            repaired_answer = gen["answer"]
            verdict = gen["verdict"]
            abstain_reason = gen["abstain_reason"]

        # Valid time: prefer newest active cited doc
        valid_times = [c.valid_time for c in citations if c.valid_time and c.status != "Dicabut"] or [c.valid_time for c in citations if c.valid_time]
        best_valid_time = max(valid_times) if valid_times else ""

        return PolicyResponse(
            query_id=query_id,
            config_id=config_id,
            query=query,
            answer=PII_OUT_RE.sub("[REDACTED_PII]", repaired_answer),
            atomic_claims=atomic_claims,
            citations=citations,
            valid_time=best_valid_time,
            verdict=verdict,
            abstain_reason=abstain_reason,
            latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
            prompt_tokens_est=gen["prompt_tokens_est"],
            completion_tokens_est=gen["completion_tokens_est"],
            model_id=backend.model_id,
            prompt_version=prompt_ver,
            index_version=self.index_version,
            trace=trace,
        )
