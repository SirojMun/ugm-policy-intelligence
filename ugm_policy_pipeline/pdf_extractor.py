"""PyMuPDF extraction, scanned-PDF detection, legal hierarchy Markdown formatting, and PII scanner."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from ugm_policy_pipeline.parsers import (
    clean_whitespace,
    extract_related_regulations,
    parse_indonesian_date,
)

# Structural boundaries that must NEVER be merged into the preceding paragraph
STRUCTURAL_START_RE = re.compile(
    r"^(?:"
    r"BAB\s+[IVXLCDM]+\b"
    r"|Bagian\s+(?:Ke(?:satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)|Pertama|[A-Za-z]+)"
    r"|Paragraf\s+\d+"
    r"|Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+)\b"
    r"|MENIMBANG\b|Menimbang\b"
    r"|MENGINGAT\b|Mengingat\b"
    r"|MEMUTUSKAN\b"
    r"|MENETAPKAN\b|Menetapkan\b"
    r"|LAMPIRAN\b|Lampiran\b"
    r"|PENJELASAN\b"
    r"|Ditetapkan\s+di\b|Diundangkan\s+di\b"
    r"|REKTOR\s+UNIVERSITAS\b|MAJELIS\s+WALI\s+AMANAT\b|SENAT\s+AKADEMIK\b"
    r"|\(\d+[a-z]?\)\s*"
    r"|[a-z]\.\s*"
    r"|\d+\.\s+"
    r"|[-•]\s+"
    r")"
)

PAGE_NUMBER_RE = re.compile(
    r"^(?:"
    r"[-–—]\s*\d+\s*[-–—]"
    r"|\d{1,3}"
    r"|(?:Halaman|Hal\.?|Page)\s*\d+(?:\s*(?:dari|of|/)\s*\d+)?"
    r"|SALINAN"
    r")$",
    re.IGNORECASE,
)

# PII Regex patterns (carefully tuned to flag mass PII while minimizing false positives)
NIK_RE = re.compile(r"\b(?:3[1-6]|1[1-9]|21|5[1-3]|6[1-5]|7[1-6]|8[12]|9[124])\d{4}(?:[0-6]\d|7[01])(?:0[1-9]|1[0-2])\d{6}\b")
NIP_RE = re.compile(r"\b(?:19|20)\d{6}(?:19|20)\d{4}[12]\d{3}\b")
UGM_NIM_RE = re.compile(r"\b\d{2}/\d{6}/[A-Z]{2,4}/\d{5}\b")
BANK_RE = re.compile(
    r"(?:rekening|no\.?\s*rek(?:ening)?|norek|bank\s+(?:bri|bni|mandiri|bca|bsi|btn|diy))\s*[:\-]?\s*\d[\d\-\s]{7,20}\d",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


@dataclass
class ExtractionResult:
    """Structured result of extracting and cleaning one PDF file."""

    is_scanned: bool
    page_count: int
    char_count: int
    chars_per_page: float
    markdown_text: str
    effective_date: str
    related_regulations: list[str]
    status_cue: str
    pii_flagged: bool
    pii_summary: dict[str, Any]


def _detect_repeated_margin_lines(pages_lines: list[list[str]]) -> set[str]:
    """Identify recurring header/footer lines appearing across >=40% of pages (min 3 pages)."""
    n_pages = len(pages_lines)
    if n_pages < 3:
        return set()

    counts: Counter[str] = Counter()
    for lines in pages_lines:
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        if not non_empty:
            continue
        candidates = set(non_empty[:3] + non_empty[-3:])
        for cand in candidates:
            if len(cand) >= 4 and not re.match(r"^(?:BAB|Pasal|Bagian|\(\d+\))", cand, re.IGNORECASE):
                counts[cand] += 1

    threshold = max(3, int(n_pages * 0.4))
    return {line for line, cnt in counts.items() if cnt >= threshold}


def _clean_page_lines(raw_pages: list[str]) -> list[str]:
    """Strip page numbers and repeated headers/footers across all PDF pages."""
    pages_lines = [p.splitlines() for p in raw_pages]
    repeated = _detect_repeated_margin_lines(pages_lines)
    cleaned_lines: list[str] = []

    for lines in pages_lines:
        # Trim leading/trailing page numbers on each page
        non_empty_indices = [i for i, l in enumerate(lines) if l.strip()]
        skip_indices: set[int] = set()
        if non_empty_indices:
            for idx in non_empty_indices[:2] + non_empty_indices[-2:]:
                stripped = lines[idx].strip()
                if PAGE_NUMBER_RE.match(stripped) or stripped in repeated:
                    skip_indices.add(idx)

        for idx, line in enumerate(lines):
            if idx in skip_indices:
                continue
            stripped = re.sub(r"[ \t]+", " ", line).strip()
            if stripped in repeated:
                continue
            if PAGE_NUMBER_RE.match(stripped):
                continue
            # Collapse duplicate consecutive blank lines without forcing breaks at page boundaries
            if stripped == "" and cleaned_lines and cleaned_lines[-1] == "":
                continue
            cleaned_lines.append(stripped)

    return cleaned_lines


def _merge_isolated_markers_and_unwrap(lines: list[str]) -> list[str]:
    """Rejoin isolated markers like '(1)' or 'a.' and unwrap mid-sentence line breaks."""
    # Step 1: Join isolated '(1)', 'a.', '1.', 'Menimbang :', 'Mengingat :' with following non-empty text
    step1: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if re.match(r"^(?:\(\d+[a-z]?\)|[a-z]\.|\d+\.)$", cur):
            j = i + 1
            while j < len(lines) and lines[j] == "":
                j += 1
            if j < len(lines) and not STRUCTURAL_START_RE.match(lines[j]):
                step1.append(f"{cur} {lines[j]}")
                i = j + 1
                continue
        if re.match(r"^(?:Menimbang|Mengingat|Menetapkan)\s*$", cur, re.IGNORECASE):
            j = i + 1
            while j < len(lines) and lines[j] == "":
                j += 1
            if j < len(lines) and lines[j].startswith(":"):
                step1.append(f"{cur} {lines[j]}")
                i = j + 1
                continue
        step1.append(cur)
        i += 1

    # Step 2: Unwrap justified mid-sentence line breaks
    merged: list[str] = []
    for ln in step1:
        if ln == "":
            if merged and merged[-1] != "":
                merged.append("")
            continue

        if not merged or merged[-1] == "":
            merged.append(ln)
            continue

        prev = merged[-1]

        # Rejoin inline cross-references broken before '(1)' or 'a.' e.g. 'pada ayat\n(1)'
        if re.search(r"\b(?:ayat|huruf|angka|butir)\s*$", prev, re.IGNORECASE) and re.match(
            r"^(?:\(\d+[a-z]?\)|[a-z]\.|\d+\.)", ln
        ):
            merged[-1] = f"{prev} {ln}"
            continue

        if STRUCTURAL_START_RE.match(ln):
            merged.append(ln)
            continue

        # Keep uppercase legal heading tags separated from their title/body lines
        prev_is_heading_tag = bool(
            re.match(
                r"^(?:BAB\s+[IVXLCDM]+|Bagian\s+\w+|Paragraf\s+\d+|Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+)|LAMPIRAN(?:\s+[IVXLCDM\d]+)?|PENJELASAN)$",
                prev,
                re.IGNORECASE,
            )
        )
        if prev_is_heading_tag:
            merged.append(ln)
            continue

        # Join broken sentence across lines/pages when prev does not terminate a sentence/clause
        if not re.search(r"[.;:!?\"”]$", prev):
            if prev.endswith("-") and len(prev) > 2 and prev[-2].isalpha():
                merged[-1] = prev[:-1] + ln
            else:
                merged[-1] = f"{prev} {ln}"
        else:
            merged.append(ln)

    return merged


def format_legal_markdown(title: str, raw_pages: list[str]) -> str:
    """Convert raw PDF page texts into hierarchical legal Markdown (#, ##, ###)."""
    cleaned_lines = _clean_page_lines(raw_pages)
    lines = _merge_isolated_markers_and_unwrap(cleaned_lines)

    out: list[str] = [f"# {clean_whitespace(title)}", ""]
    n = len(lines)
    i = 0
    seen_body_start = False

    while i < n:
        ln = lines[i].strip()
        if not ln:
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        # Skip redundant cover title banner lines right before MENIMBANG / BAB I if near start
        if not seen_body_start and i < 12 and re.match(
            r"^(?:REKTOR|MAJELIS WALI AMANAT|SENAT AKADEMIK)\s+UNIVERSITAS GADJAH MADA$",
            ln,
            re.IGNORECASE,
        ):
            i += 1
            continue

        # Major section: MENIMBANG / MENGINGAT / MEMUTUSKAN
        m_kons = re.match(r"^(Menimbang|Mengingat)\s*:\s*(.*)$", ln, re.IGNORECASE)
        if m_kons:
            seen_body_start = True
            sec_name = m_kons.group(1).upper()
            rest = m_kons.group(2).strip()
            out.extend(["", f"## {sec_name}", ""])
            if rest:
                out.append(rest)
            i += 1
            continue

        if re.match(r"^MEMUTUSKAN\s*:?$", ln, re.IGNORECASE):
            seen_body_start = True
            out.extend(["", "## MEMUTUSKAN", ""])
            i += 1
            continue

        m_menetapkan = re.match(r"^Menetapkan\s*:\s*(.*)$", ln, re.IGNORECASE)
        if m_menetapkan:
            seen_body_start = True
            out.extend(["", "## MENETAPKAN", ""])
            if m_menetapkan.group(1).strip():
                out.append(m_menetapkan.group(1).strip())
            i += 1
            continue

        # Major section: BAB I, BAB II, ... (combine optional uppercase subtitle on next line)
        m_bab = re.match(r"^(BAB\s+[IVXLCDM]+)\b(.*)$", ln)
        if m_bab:
            seen_body_start = True
            bab_head = m_bab.group(1).strip()
            tail = m_bab.group(2).strip(" :—-")
            if not tail and i + 1 < n and lines[i + 1].strip() and not STRUCTURAL_START_RE.match(lines[i + 1].strip()):
                tail = lines[i + 1].strip()
                i += 1
            heading = f"## {bab_head} — {tail}" if tail else f"## {bab_head}"
            out.extend(["", heading, ""])
            i += 1
            continue

        # Major section: LAMPIRAN / PENJELASAN
        if re.match(r"^(LAMPIRAN|PENJELASAN)\b", ln):
            seen_body_start = True
            out.extend(["", f"## {ln}", ""])
            i += 1
            continue

        # Sub-section: Bagian Kesatu / Paragraf 1 (combine optional title on next line)
        m_bagian = re.match(
            r"^((?:Bagian\s+(?:Ke(?:satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)|Pertama|[A-Za-z]+)|Paragraf\s+\d+))\b(.*)$",
            ln,
            re.IGNORECASE,
        )
        if m_bagian:
            seen_body_start = True
            sub_head = m_bagian.group(1).strip()
            tail = m_bagian.group(2).strip(" :—-")
            if not tail and i + 1 < n and lines[i + 1].strip() and not STRUCTURAL_START_RE.match(lines[i + 1].strip()):
                tail = lines[i + 1].strip()
                i += 1
            heading = f"### {sub_head} — {tail}" if tail else f"### {sub_head}"
            out.extend(["", heading, ""])
            i += 1
            continue

        # Sub-section: Pasal 1, Pasal 10B, Pasal I
        m_pasal = re.match(r"^(Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+))\b(.*)$", ln)
        if m_pasal:
            seen_body_start = True
            pasal_head = m_pasal.group(1).strip()
            tail = m_pasal.group(2).strip()
            out.extend(["", f"### {pasal_head}", ""])
            if tail:
                out.append(tail)
            i += 1
            continue

        out.append(ln)
        i += 1

    # Collapse >2 consecutive blank lines
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def scan_pii(text: str, thresholds: dict[str, int] | None = None) -> tuple[bool, dict[str, Any]]:
    """Scan extracted text for mass PII patterns (16-digit NIK, bank accounts, NIM/NIP lists)."""
    cfg = thresholds or {
        "nik_count_flag": 3,
        "nip_count_flag": 5,
        "nim_count_flag": 5,
        "bank_account_count_flag": 2,
        "email_count_flag": 10,
    }
    niks = set(NIK_RE.findall(text))
    nips = set(NIP_RE.findall(text))
    nims = set(UGM_NIM_RE.findall(text))
    banks = BANK_RE.findall(text)
    emails = set(EMAIL_RE.findall(text))

    counts = {
        "nik_count": len(niks),
        "nip_count": len(nips),
        "nim_count": len(nims),
        "bank_account_count": len(banks),
        "email_count": len(emails),
    }

    reasons: list[str] = []
    if counts["nik_count"] >= cfg.get("nik_count_flag", 3):
        reasons.append(f"nik>={cfg['nik_count_flag']} ({counts['nik_count']})")
    if counts["nip_count"] >= cfg.get("nip_count_flag", 5):
        reasons.append(f"nip>={cfg['nip_count_flag']} ({counts['nip_count']})")
    if counts["nim_count"] >= cfg.get("nim_count_flag", 5):
        reasons.append(f"nim>={cfg['nim_count_flag']} ({counts['nim_count']})")
    if counts["bank_account_count"] >= cfg.get("bank_account_count_flag", 2):
        reasons.append(f"bank>={cfg['bank_account_count_flag']} ({counts['bank_account_count']})")
    if counts["email_count"] >= cfg.get("email_count_flag", 10):
        reasons.append(f"email>={cfg['email_count_flag']} ({counts['email_count']})")

    flagged = len(reasons) > 0
    return flagged, {**counts, "reasons": reasons}


def _extract_in_text_legal_signals(text: str) -> tuple[str, list[str], str]:
    """Extract effective date, related regulations, and repeal/amendment signals from body text."""
    effective_date = ""
    m_eff = re.search(
        r"(?:mulai\s+berlaku\s+pada\s+tanggal|Ditetapkan\s+di\s+Yogyakarta\s+pada\s+tanggal)\s+(\d{1,2}\s+[A-Za-z]+\s+(?:19|20)\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    if m_eff:
        effective_date = parse_indonesian_date(m_eff.group(1))

    related = extract_related_regulations(text)
    status_cue = ""
    if re.search(r"dicabut\s+dan\s+dinyatakan\s+tidak\s+berlaku", text, flags=re.IGNORECASE):
        status_cue = "repeals_other"
    elif re.search(r"diubah\s+dengan\s+Peraturan", text, flags=re.IGNORECASE):
        status_cue = "amends_other"

    return effective_date, related, status_cue


def extract_pdf_document(
    pdf_path: str | Path,
    title: str,
    min_chars_per_page: int = 200,
    pii_thresholds: dict[str, int] | None = None,
) -> ExtractionResult:
    """Open a PDF with PyMuPDF, detect scanned docs (<200 chars/page), extract Markdown & PII scan."""
    with pymupdf.open(str(pdf_path)) as doc:
        page_count = len(doc)
        raw_pages = [page.get_text("text") or "" for page in doc]

    # Measure raw digital character density cleanly (whitespace normalized)
    raw_char_count = sum(len(re.sub(r"\s+", " ", p).strip()) for p in raw_pages)
    chars_per_page = raw_char_count / max(1, page_count)

    if page_count == 0 or chars_per_page < min_chars_per_page:
        return ExtractionResult(
            is_scanned=True,
            page_count=page_count,
            char_count=raw_char_count,
            chars_per_page=round(chars_per_page, 2),
            markdown_text="",
            effective_date="",
            related_regulations=[],
            status_cue="",
            pii_flagged=False,
            pii_summary={},
        )

    markdown_text = format_legal_markdown(title=title, raw_pages=raw_pages)
    clean_char_count = len(markdown_text)
    effective_date, related, status_cue = _extract_in_text_legal_signals(markdown_text)
    pii_flagged, pii_summary = scan_pii(markdown_text, thresholds=pii_thresholds)

    return ExtractionResult(
        is_scanned=False,
        page_count=page_count,
        char_count=clean_char_count,
        chars_per_page=round(chars_per_page, 2),
        markdown_text=markdown_text,
        effective_date=effective_date,
        related_regulations=related,
        status_cue=status_cue,
        pii_flagged=pii_flagged,
        pii_summary=pii_summary,
    )
