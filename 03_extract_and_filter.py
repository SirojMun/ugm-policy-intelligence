#!/usr/bin/env python3
"""Stage 3: Scanned-PDF filtering, legal hierarchy Markdown extraction, PII audit & dedup."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ugm_policy_pipeline.io_utils import (
    SOURCE_REGISTER_FIELDS,
    ensure_dirs,
    load_config,
    load_jsonl,
    sha256_file,
    sha256_text,
    write_csv,
    write_jsonl,
)
from ugm_policy_pipeline.pdf_extractor import extract_pdf_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("03_extract_and_filter")


def _norm_reg_ref(doc_type: str, reg_num: str, year: str | int) -> str:
    """Create canonical lookup key for matching repeal/amendment cross-references."""
    num_m = re.match(r"(\d+)", str(reg_num or ""))
    yr_m = re.search(r"((?:19|20)\d{2})", str(year or "") + " " + str(reg_num or ""))
    if not num_m or not yr_m:
        return ""
    return f"{doc_type.lower()}::{int(num_m.group(1))}::{yr_m.group(1)}"


def apply_cross_document_status(records: list[dict[str, Any]]) -> None:
    """Update status to Dicabut/Diubah when a newer corpus document explicitly repeals/amends it."""
    by_key: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = _norm_reg_ref(rec.get("doc_type", ""), rec.get("regulation_number", ""), rec.get("year", ""))
        if key:
            by_key[key] = rec

    for rec in records:
        md_path = Path(rec.get("markdown_path") or "")
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8", errors="replace")

        for m in re.finditer(
            r"(Peraturan\s+(?:Rektor|MWA|Majelis\s+Wali\s+Amanat|Senat\s+Akademik)[^.\n]{0,160}?Nomor\s+(\d+)[^.\n]{0,40}?Tahun\s+((?:19|20)\d{2})[^.\n]{0,120}?dicabut\s+dan\s+dinyatakan\s+tidak\s+berlaku)",
            text,
            flags=re.IGNORECASE,
        ):
            dtype = "Peraturan Rektor" if "rektor" in m.group(1).lower() else (
                "Peraturan MWA" if "mwa" in m.group(1).lower() or "wali amanat" in m.group(1).lower() else "Peraturan Senat Akademik"
            )
            key = f"{dtype.lower()}::{int(m.group(2))}::{m.group(3)}"
            target = by_key.get(key)
            if target and target["doc_id"] != rec["doc_id"]:
                target["status"] = "Dicabut"

        for m in re.finditer(
            r"Perubahan\s+(?:Kedua\s+|Ketiga\s+|Keempat\s+)?atas\s+(Peraturan\s+(?:Rektor|MWA|Majelis\s+Wali\s+Amanat|Senat\s+Akademik)[^.\n]{0,80}?Nomor\s+(\d+)[^.\n]{0,30}?Tahun\s+((?:19|20)\d{2}))",
            rec.get("title", "") + "\n" + text[:2500],
            flags=re.IGNORECASE,
        ):
            dtype = "Peraturan Rektor" if "rektor" in m.group(1).lower() else (
                "Peraturan MWA" if "mwa" in m.group(1).lower() or "wali amanat" in m.group(1).lower() else "Peraturan Senat Akademik"
            )
            key = f"{dtype.lower()}::{int(m.group(2))}::{m.group(3)}"
            target = by_key.get(key)
            if target and target["doc_id"] != rec["doc_id"] and target.get("status") in ("Unknown", "Berlaku"):
                target["status"] = "Diubah"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3: Filter scanned PDFs, extract hierarchical Markdown, scan PII, and deduplicate."
    )
    parser.add_argument("--config", default="sources_config.yaml", help="Path to YAML config file.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if markdown file already exists.")
    parser.add_argument(
        "--prune-excluded-pdfs",
        action="store_true",
        help="Delete raw PDF files on disk for excluded_scanned and excluded_duplicate_* entries after logging.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    paths = cfg["paths"]

    records = load_jsonl(paths["source_register_jsonl"])
    if not records:
        logger.error("No downloaded records in %s. Run 02_download_and_hash.py first.", paths["source_register_jsonl"])
        return 1

    min_chars = int(cfg["filtering"].get("min_chars_per_page", 200))
    pii_thresholds = cfg.get("pii_thresholds", {})
    corpus_md_dir = Path(paths["corpus_md_dir"])

    scanned_logs: list[dict[str, Any]] = []
    pii_logs: list[dict[str, Any]] = []
    dedup_logs: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    all_registered: list[dict[str, Any]] = []

    seen_pdf_sha: dict[str, str] = {}
    seen_text_sha: dict[str, str] = {}
    canonical_paths: dict[str, Path] = {}

    def _prune_if_enabled(rec_obj: dict[str, Any], candidate_path: Path) -> None:
        if not args.prune_excluded_pdfs or not candidate_path.exists():
            return
        # Never delete a file path shared by a canonical kept document
        if candidate_path in canonical_paths.values():
            return
        candidate_path.unlink(missing_ok=True)
        rec_obj["local_pdf_path"] = ""

    for rec in records:
        doc_id = rec["doc_id"]
        if str(rec.get("corpus_status", "")).startswith("excluded_") and not args.force:
            # Already excluded in a previous Stage 3 run; prune raw file if still present
            old_pdf = Path(rec.get("local_pdf_path") or "")
            if old_pdf.is_file():
                _prune_if_enabled(rec, old_pdf)
            if rec.get("corpus_status") == "excluded_scanned":
                scanned_logs.append(
                    {
                        "doc_id": doc_id,
                        "title": rec.get("title", ""),
                        "pdf_url": rec.get("pdf_url", ""),
                        "page_count": rec.get("page_count", 0),
                        "char_count": rec.get("char_count", 0),
                        "chars_per_page": rec.get("chars_per_page", 0.0),
                        "reason": f"chars_per_page_below_{min_chars}",
                    }
                )
            elif rec.get("corpus_status") in ("excluded_duplicate_pdf", "excluded_duplicate_text"):
                dedup_logs.append(
                    {
                        "duplicate_doc_id": doc_id,
                        "canonical_doc_id": seen_pdf_sha.get(rec.get("sha256_pdf", ""), "")
                        or seen_text_sha.get(rec.get("sha256_text", ""), ""),
                        "dedup_type": "sha256_pdf" if rec["corpus_status"] == "excluded_duplicate_pdf" else "sha256_text",
                        "hash": rec.get("sha256_pdf") or rec.get("sha256_text") or "",
                        "title": rec.get("title", ""),
                    }
                )
            all_registered.append(rec)
            continue

        pdf_path = Path(rec.get("local_pdf_path") or "")
        if not pdf_path.exists():
            logger.warning("Missing PDF on disk for %s (%s); skipping.", doc_id, pdf_path)
            rec["corpus_status"] = "missing_pdf"
            all_registered.append(rec)
            continue

        pdf_sha = rec.get("sha256_pdf") or sha256_file(pdf_path)
        rec["sha256_pdf"] = pdf_sha

        # 1. Deduplicate by sha256_pdf
        if pdf_sha in seen_pdf_sha:
            rec["corpus_status"] = "excluded_duplicate_pdf"
            dedup_logs.append(
                {
                    "duplicate_doc_id": doc_id,
                    "canonical_doc_id": seen_pdf_sha[pdf_sha],
                    "dedup_type": "sha256_pdf",
                    "hash": pdf_sha,
                    "title": rec.get("title", ""),
                }
            )
            _prune_if_enabled(rec, pdf_path)
            all_registered.append(rec)
            continue
        seen_pdf_sha[pdf_sha] = doc_id

        # Preserve prior scanned classification unless --force
        if not args.force and rec.get("corpus_status") == "excluded_scanned":
            scanned_logs.append(
                {
                    "doc_id": doc_id,
                    "title": rec.get("title", ""),
                    "pdf_url": rec.get("pdf_url", ""),
                    "page_count": rec.get("page_count", 0),
                    "char_count": rec.get("char_count", 0),
                    "chars_per_page": rec.get("chars_per_page", 0.0),
                    "reason": f"chars_per_page_below_{min_chars}",
                }
            )
            all_registered.append(rec)
            continue

        md_path = corpus_md_dir / f"{doc_id}.md"

        if not args.force and md_path.exists() and rec.get("sha256_text") and rec.get("page_count"):
            md_text = md_path.read_text(encoding="utf-8")
            text_sha = sha256_text(md_text)
            rec["sha256_text"] = text_sha
            rec["markdown_path"] = str(md_path)
        else:
            try:
                res = extract_pdf_document(
                    pdf_path=pdf_path,
                    title=rec["title"],
                    min_chars_per_page=min_chars,
                    pii_thresholds=pii_thresholds,
                )
            except Exception as exc:
                logger.warning("PyMuPDF failed on %s (%s): %s", doc_id, pdf_path, exc)
                rec["corpus_status"] = "excluded_pdf_error"
                scanned_logs.append(
                    {
                        "doc_id": doc_id,
                        "title": rec.get("title", ""),
                        "pdf_url": rec.get("pdf_url", ""),
                        "page_count": 0,
                        "char_count": 0,
                        "chars_per_page": 0.0,
                        "reason": f"pdf_open_error:{type(exc).__name__}",
                    }
                )
                all_registered.append(rec)
                continue

            rec["page_count"] = res.page_count
            rec["char_count"] = res.char_count
            rec["chars_per_page"] = res.chars_per_page

            if res.is_scanned:
                logger.info(
                    "Filtered scanned/low-text PDF %s (pages=%d, chars/page=%.1f < %d)",
                    doc_id,
                    res.page_count,
                    res.chars_per_page,
                    min_chars,
                )
                rec["corpus_status"] = "excluded_scanned"
                scanned_logs.append(
                    {
                        "doc_id": doc_id,
                        "title": rec.get("title", ""),
                        "pdf_url": rec.get("pdf_url", ""),
                        "page_count": res.page_count,
                        "char_count": res.char_count,
                        "chars_per_page": res.chars_per_page,
                        "reason": f"chars_per_page_below_{min_chars}",
                    }
                )
                _prune_if_enabled(rec, pdf_path)
                all_registered.append(rec)
                continue

            text_sha = sha256_text(res.markdown_text)

            # 2. Deduplicate by sha256_text before writing Markdown
            if text_sha in seen_text_sha:
                rec["sha256_text"] = text_sha
                rec["corpus_status"] = "excluded_duplicate_text"
                dedup_logs.append(
                    {
                        "duplicate_doc_id": doc_id,
                        "canonical_doc_id": seen_text_sha[text_sha],
                        "dedup_type": "sha256_text",
                        "hash": text_sha,
                        "title": rec.get("title", ""),
                    }
                )
                _prune_if_enabled(rec, pdf_path)
                all_registered.append(rec)
                continue
            seen_text_sha[text_sha] = doc_id

            md_path.write_text(res.markdown_text, encoding="utf-8")

            eff = rec.get("effective_date") or res.effective_date
            rec["effective_date"] = eff
            if eff:
                rec["valid_time"] = eff
            elif not rec.get("valid_time") and rec.get("year"):
                rec["valid_time"] = str(rec["year"])

            own_reg = (rec.get("regulation_number") or "").lower()
            merged_related = [
                r
                for r in dict.fromkeys((rec.get("related_regulations") or []) + res.related_regulations)
                if not (own_reg and own_reg in r.lower() and rec.get("doc_type", "").lower() in r.lower())
            ]
            rec["related_regulations"] = merged_related
            rec["sha256_text"] = text_sha
            rec["pii_flagged"] = res.pii_flagged
            rec["pii_summary"] = res.pii_summary
            rec["markdown_path"] = str(md_path)

        # Handle cached Markdown text deduplication
        if rec["sha256_text"] not in seen_text_sha:
            seen_text_sha[rec["sha256_text"]] = doc_id
        elif seen_text_sha[rec["sha256_text"]] != doc_id:
            rec["corpus_status"] = "excluded_duplicate_text"
            dedup_logs.append(
                {
                    "duplicate_doc_id": doc_id,
                    "canonical_doc_id": seen_text_sha[rec["sha256_text"]],
                    "dedup_type": "sha256_text",
                    "hash": rec["sha256_text"],
                    "title": rec.get("title", ""),
                }
            )
            _prune_if_enabled(rec, pdf_path)
            all_registered.append(rec)
            continue

        rec["corpus_status"] = "clean"
        if pdf_path.is_file():
            canonical_paths[doc_id] = pdf_path
        if rec.get("pii_flagged"):
            pii_logs.append(
                {
                    "doc_id": doc_id,
                    "title": rec.get("title", ""),
                    "source_url": rec.get("source_url", ""),
                    "pdf_url": rec.get("pdf_url", ""),
                    "pii_summary": rec.get("pii_summary", {}),
                }
            )

        valid_records.append(rec)
        all_registered.append(rec)

    apply_cross_document_status(valid_records)

    clean_corpus: list[dict[str, Any]] = []
    for rec in valid_records:
        md_text = Path(rec["markdown_path"]).read_text(encoding="utf-8")
        clean_corpus.append(
            {
                **{k: rec.get(k, "") for k in SOURCE_REGISTER_FIELDS},
                "text": md_text,
            }
        )

    # Keep all downloaded records in source_register (with corpus_status) for full audit & checkpoint resume
    write_jsonl(paths["source_register_jsonl"], all_registered)
    write_csv(paths["source_register_csv"], all_registered, SOURCE_REGISTER_FIELDS)
    write_jsonl(paths["corpus_clean_jsonl"], clean_corpus)

    write_csv(
        paths["skipped_scanned_csv"],
        scanned_logs,
        ["doc_id", "title", "pdf_url", "page_count", "char_count", "chars_per_page", "reason"],
    )
    write_csv(
        paths["flagged_pii_csv"],
        pii_logs,
        ["doc_id", "title", "source_url", "pdf_url", "pii_summary"],
    )
    write_csv(
        paths["deduplicated_csv"],
        dedup_logs,
        ["duplicate_doc_id", "canonical_doc_id", "dedup_type", "hash", "title"],
    )

    stats = {
        "total_downloaded_registered": len(all_registered),
        "total_clean_corpus": len(valid_records),
        "skipped_scanned_pdfs": len(scanned_logs),
        "deduplicated_removed": len(dedup_logs),
        "pii_flagged_for_review": len(pii_logs),
        "by_doc_type": dict(Counter(r.get("doc_type", "Unknown") for r in valid_records)),
        "by_year": dict(sorted(Counter(str(r.get("year") or "Unknown") for r in valid_records).items())),
        "by_status": dict(Counter(r.get("status", "Unknown") for r in valid_records)),
        "total_pages": sum(int(r.get("page_count") or 0) for r in valid_records),
        "total_chars": sum(int(r.get("char_count") or 0) for r in valid_records),
    }
    Path(paths["corpus_stats_json"]).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Stage 3 complete! Corpus stats:\n%s", json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
