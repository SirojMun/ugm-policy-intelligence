"""Ablation A1: Fixed vs. Heading-Aware vs. Hierarchical Legal Markdown Chunking."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from src.schemas import ChunkRecord

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
PASAL_RE = re.compile(r"\b(Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+))\b")


def _parse_year(val: Any) -> int | None:
    if val is None or val == "":
        return None
    m = re.search(r"((?:19|20)\d{2})", str(val))
    return int(m.group(1)) if m else None


def _split_words_window(words: list[str], max_words: int, overlap_words: int) -> list[str]:
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max(1, max_words - overlap_words)
    windows: list[str] = []
    for start in range(0, len(words), step):
        seg = words[start : start + max_words]
        if not seg:
            break
        windows.append(" ".join(seg))
        if start + max_words >= len(words):
            break
    return windows


def _extract_sections(md_text: str) -> list[dict[str, Any]]:
    """Parse Markdown into ordered structural sections preserving H1/H2/H3 breadcrumb."""
    h1 = ""
    h2 = ""
    h3 = ""
    sections: list[dict[str, Any]] = []
    buf: list[str] = []

    def _flush() -> None:
        body = "\n".join(buf).strip()
        if not body:
            return
        parts = [p for p in (h1, h2, h3) if p]
        section_path = " > ".join(parts)
        pasal_m = PASAL_RE.search(h3 or h2)
        sections.append(
            {
                "h1": h1,
                "h2": h2,
                "h3": h3,
                "section_path": section_path,
                "pasal": pasal_m.group(1) if pasal_m else "",
                "body": body,
            }
        )

    for line in md_text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            _flush()
            buf = []
            level = len(m.group(1))
            heading_txt = m.group(2).strip()
            if level == 1:
                h1, h2, h3 = heading_txt, "", ""
            elif level == 2:
                h2, h3 = heading_txt, ""
            else:
                h3 = heading_txt
        else:
            buf.append(line)
    _flush()

    if not sections and md_text.strip():
        sections.append(
            {
                "h1": "",
                "h2": "",
                "h3": "",
                "section_path": "",
                "pasal": "",
                "body": md_text.strip(),
            }
        )
    return sections


def chunk_fixed(doc: dict[str, Any], max_words: int = 320, overlap_words: int = 64) -> list[ChunkRecord]:
    """Strategy 1 (A1): Fixed word-window chunking ignoring legal heading boundaries."""
    text = (doc.get("text") or "").strip()
    words = text.split()
    windows = _split_words_window(words, max_words=max_words, overlap_words=overlap_words)
    year = _parse_year(doc.get("year"))
    chunks: list[ChunkRecord] = []
    for idx, win in enumerate(windows, start=1):
        pasal_m = PASAL_RE.search(win)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc['doc_id']}__fixed_{idx:04d}",
                doc_id=doc["doc_id"],
                strategy="fixed",
                level="single",
                title=doc["title"],
                doc_type=doc.get("doc_type", "Unknown"),
                year=year,
                valid_time=str(doc.get("valid_time") or ""),
                status=doc.get("status", "Unknown"),
                version=doc.get("version", "original"),
                usage_rights=doc.get("usage_rights", "Public Academic Regulation - UGM"),
                section_path=doc["title"],
                pasal=pasal_m.group(1) if pasal_m else "",
                word_count=len(win.split()),
                char_count=len(win),
                text=win,
            )
        )
    return chunks


def chunk_heading_aware(doc: dict[str, Any], max_words: int = 380, overlap_words: int = 40) -> list[ChunkRecord]:
    """Strategy 2 (A1): Splits along #, ##, ### Markdown headings with section breadcrumbs."""
    sections = _extract_sections(doc.get("text") or "")
    year = _parse_year(doc.get("year"))
    chunks: list[ChunkRecord] = []
    seq = 0

    for sec in sections:
        header_prefix = f"[{sec['section_path']}]\n" if sec["section_path"] else ""
        words = sec["body"].split()
        windows = _split_words_window(words, max_words=max_words, overlap_words=overlap_words)
        for win in windows:
            seq += 1
            full_text = f"{header_prefix}{win}".strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc['doc_id']}__head_{seq:04d}",
                    doc_id=doc["doc_id"],
                    strategy="heading_aware",
                    level="single",
                    title=doc["title"],
                    doc_type=doc.get("doc_type", "Unknown"),
                    year=year,
                    valid_time=str(doc.get("valid_time") or ""),
                    status=doc.get("status", "Unknown"),
                    version=doc.get("version", "original"),
                    usage_rights=doc.get("usage_rights", "Public Academic Regulation - UGM"),
                    section_path=sec["section_path"] or doc["title"],
                    pasal=sec["pasal"],
                    word_count=len(full_text.split()),
                    char_count=len(full_text),
                    text=full_text,
                )
            )
    return chunks


def chunk_hierarchical(
    doc: dict[str, Any],
    child_max_words: int = 240,
    parent_max_words: int = 900,
) -> list[ChunkRecord]:
    """Strategy 3 (A1): Parent-child legal chunking (BAB/major section parent -> Pasal/Ayat child)."""
    sections = _extract_sections(doc.get("text") or "")
    year = _parse_year(doc.get("year"))
    chunks: list[ChunkRecord] = []

    # Group sections by H2 (BAB / MENIMBANG / MENGINGAT / LAMPIRAN)
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    for sec in sections:
        group_key = sec["h2"] or sec["h1"] or doc["title"]
        if grouped and grouped[-1][0] == group_key:
            grouped[-1][1].append(sec)
        else:
            grouped.append((group_key, [sec]))

    p_seq = 0
    c_seq = 0
    for group_key, sec_list in grouped:
        parent_text_full = "\n\n".join(
            f"{('### ' + s['h3'] + '\n') if s['h3'] else ''}{s['body']}" for s in sec_list
        ).strip()
        parent_windows = _split_words_window(parent_text_full.split(), max_words=parent_max_words, overlap_words=60)
        parent_ids: list[str] = []
        for pw in parent_windows:
            p_seq += 1
            pid = f"{doc['doc_id']}__hier_p_{p_seq:04d}"
            parent_ids.append(pid)
            p_full = f"[{doc['title']} > {group_key}]\n{pw}".strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=pid,
                    doc_id=doc["doc_id"],
                    parent_id=pid,
                    strategy="hierarchical",
                    level="parent",
                    title=doc["title"],
                    doc_type=doc.get("doc_type", "Unknown"),
                    year=year,
                    valid_time=str(doc.get("valid_time") or ""),
                    status=doc.get("status", "Unknown"),
                    version=doc.get("version", "original"),
                    usage_rights=doc.get("usage_rights", "Public Academic Regulation - UGM"),
                    section_path=f"{doc['title']} > {group_key}",
                    pasal="",
                    word_count=len(p_full.split()),
                    char_count=len(p_full),
                    text=p_full,
                )
            )

        default_pid = parent_ids[0] if parent_ids else ""
        for sec in sec_list:
            # Split long Pasal along ayat/paragraphs when needed
            sec_words = sec["body"].split()
            child_windows = _split_words_window(sec_words, max_words=child_max_words, overlap_words=32)
            for cw in child_windows:
                c_seq += 1
                cid = f"{doc['doc_id']}__hier_c_{c_seq:04d}"
                c_full = f"[{sec['section_path']}]\n{cw}".strip() if sec["section_path"] else cw
                chunks.append(
                    ChunkRecord(
                        chunk_id=cid,
                        doc_id=doc["doc_id"],
                        parent_id=default_pid,
                        strategy="hierarchical",
                        level="child",
                        title=doc["title"],
                        doc_type=doc.get("doc_type", "Unknown"),
                        year=year,
                        valid_time=str(doc.get("valid_time") or ""),
                        status=doc.get("status", "Unknown"),
                        version=doc.get("version", "original"),
                        usage_rights=doc.get("usage_rights", "Public Academic Regulation - UGM"),
                        section_path=sec["section_path"] or doc["title"],
                        pasal=sec["pasal"],
                        word_count=len(c_full.split()),
                        char_count=len(c_full),
                        text=c_full,
                    )
                )

    return chunks


def chunk_corpus(
    docs: list[dict[str, Any]],
    strategy: Literal["fixed", "heading_aware", "hierarchical"],
) -> list[ChunkRecord]:
    """Chunk an entire corpus list using the selected A1 strategy."""
    fn = {
        "fixed": chunk_fixed,
        "heading_aware": chunk_heading_aware,
        "hierarchical": chunk_hierarchical,
    }[strategy]
    all_chunks: list[ChunkRecord] = []
    for doc in docs:
        all_chunks.extend(fn(doc))
    return all_chunks


def compute_chunk_stats(chunks: list[ChunkRecord]) -> dict[str, Any]:
    """Compute index size and context length metrics required by Ablation A1."""
    if not chunks:
        return {"total_chunks": 0}
    words = sorted(c.word_count for c in chunks)
    chars = sum(c.char_count for c in chunks)
    p95_idx = min(len(words) - 1, int(0.95 * len(words)))
    return {
        "strategy": chunks[0].strategy,
        "total_chunks": len(chunks),
        "parent_chunks": sum(1 for c in chunks if c.level == "parent"),
        "child_or_single_chunks": sum(1 for c in chunks if c.level != "parent"),
        "total_chars": chars,
        "avg_words": round(sum(words) / len(words), 2),
        "p95_words": words[p95_idx],
        "max_words": words[-1],
    }


def load_corpus(corpus_jsonl: str | Path = "corpus_clean.jsonl") -> list[dict[str, Any]]:
    """Load clean extracted documents from corpus_clean.jsonl."""
    records: list[dict[str, Any]] = []
    with Path(corpus_jsonl).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
