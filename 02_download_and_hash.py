#!/usr/bin/env python3
"""Stage 2: Resumable PDF downloader, SHA-256 hasher, and source_register builder."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ugm_policy_pipeline.http_client import EthicalHttpClient, HttpConfig
from ugm_policy_pipeline.io_utils import (
    SOURCE_REGISTER_FIELDS,
    append_csv_row,
    append_jsonl,
    ensure_dirs,
    load_config,
    load_jsonl,
    sha256_bytes,
    write_csv,
    write_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("02_download_and_hash")

ERROR_FIELDS = ["doc_id", "pdf_url", "source_url", "error_type", "detail", "timestamp"]


def _is_valid_pdf_payload(content: bytes, content_type: str) -> bool:
    """Verify downloaded payload is genuinely a PDF (prevents saving HTML/SSO error pages)."""
    if content.lstrip()[:5] == b"%PDF-":
        return True
    if "application/pdf" in (content_type or "").lower() and b"%PDF-" in content[:1024]:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2: Download PDFs, compute SHA-256, and populate source_register.")
    parser.add_argument("--config", default="sources_config.yaml", help="Path to YAML config file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of new PDFs to download.")
    parser.add_argument("--no-resume", action="store_true", help="Re-download even if present in source_register.jsonl.")
    parser.add_argument("--no-delay", action="store_true", help="Disable rate-limit sleep (only for offline/mock testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    paths = cfg["paths"]
    http_cfg = cfg["http"]

    catalog = load_jsonl(paths["catalog_filtered"])
    if not catalog:
        logger.error("No candidates found in %s. Run 01_crawl_metadata.py first.", paths["catalog_filtered"])
        return 1

    reg_jsonl_path = Path(paths["source_register_jsonl"])
    reg_csv_path = Path(paths["source_register_csv"])
    raw_pdfs_dir = Path(paths["raw_pdfs_dir"])
    errors_csv_path = Path(paths["download_errors_csv"])

    existing_records: list[dict[str, Any]] = [] if args.no_resume else load_jsonl(reg_jsonl_path)
    if args.no_resume and reg_jsonl_path.exists():
        reg_jsonl_path.unlink()

    completed_by_id: dict[str, dict[str, Any]] = {}
    completed_urls: set[str] = set()
    for rec in existing_records:
        local_path = Path(rec.get("local_pdf_path") or "")
        already_excluded = str(rec.get("corpus_status", "")).startswith("excluded_")
        if rec.get("sha256_pdf") and (already_excluded or (local_path.exists() and local_path.stat().st_size > 0)):
            completed_by_id[rec["doc_id"]] = rec
            completed_urls.add(rec["pdf_url"])

    max_pdf_bytes = int(float(http_cfg.get("max_pdf_mb", 25)) * 1024 * 1024)

    logger.info(
        "Loaded %d candidate docs (%d already downloaded in checkpoint).",
        len(catalog),
        len(completed_by_id),
    )

    client = EthicalHttpClient(
        HttpConfig(
            user_agent=http_cfg["user_agent"],
            min_delay_sec=float(http_cfg["min_delay_sec"]),
            max_delay_sec=float(http_cfg["max_delay_sec"]),
            timeout_sec=int(http_cfg["timeout_sec"]),
            max_retries=int(http_cfg["max_retries"]),
            backoff_base_sec=float(http_cfg["backoff_base_sec"]),
            allow_insecure_ssl_fallback=bool(http_cfg["allow_insecure_ssl_fallback"]),
            insecure_ssl_hosts=tuple(http_cfg.get("insecure_ssl_hosts", ())),
            sso_indicators=tuple(cfg["filtering"].get("sso_indicators", ())),
        ),
        disable_delay=args.no_delay,
    )

    downloaded_now = 0
    errors_now = 0

    for item in catalog:
        doc_id = item["doc_id"]
        pdf_url = item["pdf_url"]

        if doc_id in completed_by_id or pdf_url in completed_urls:
            continue
        if args.limit is not None and downloaded_now >= args.limit:
            break

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            resp = client.get(pdf_url, check_robots=True, stream=True)
            ct = resp.headers.get("Content-Type", "")
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > max_pdf_bytes:
                resp.close()
                raise ValueError(f"PDF exceeds max_pdf_mb ({int(cl) / (1024 * 1024):.1f} MB)")
            payload = resp.content
            resp.close()
            if len(payload) > max_pdf_bytes:
                raise ValueError(f"PDF exceeds max_pdf_mb ({len(payload) / (1024 * 1024):.1f} MB)")
            if not _is_valid_pdf_payload(payload, ct):
                raise ValueError(f"Non-PDF response (Content-Type={ct}, bytes={len(payload)})")

            pdf_sha = sha256_bytes(payload)
            local_pdf = raw_pdfs_dir / f"{doc_id}.pdf"
            local_pdf.write_bytes(payload)

            record = {
                "doc_id": doc_id,
                "title": item["title"],
                "doc_type": item["doc_type"],
                "regulation_number": item.get("regulation_number", ""),
                "year": item.get("year") or "",
                "valid_time": item.get("valid_time", ""),
                "effective_date": item.get("effective_date", ""),
                "status": item.get("status", "Unknown"),
                "related_regulations": item.get("related_regulations", []),
                "version": item.get("version", "original"),
                "usage_rights": item.get("usage_rights", cfg["project"]["usage_rights"]),
                "source_url": item.get("source_url", ""),
                "pdf_url": pdf_url,
                "access_timestamp": now_utc,
                "sha256_pdf": pdf_sha,
                "sha256_text": "",
                "page_count": "",
                "char_count": "",
                "chars_per_page": "",
                "pii_flagged": False,
                "pii_summary": {},
                "corpus_status": "downloaded",
                "source_name": item.get("source_name", ""),
                "local_pdf_path": str(local_pdf),
                "markdown_path": "",
            }
            append_jsonl(reg_jsonl_path, record)
            completed_by_id[doc_id] = record
            completed_urls.add(pdf_url)
            downloaded_now += 1
            logger.info("[%d] Saved %s (%d bytes, sha256=%s...)", downloaded_now, doc_id, len(payload), pdf_sha[:12])

        except PermissionError as perm_err:
            errors_now += 1
            logger.warning("Skipped %s (%s): %s", doc_id, pdf_url, perm_err)
            append_csv_row(
                errors_csv_path,
                {
                    "doc_id": doc_id,
                    "pdf_url": pdf_url,
                    "source_url": item.get("source_url", ""),
                    "error_type": "sso_or_robots_blocked",
                    "detail": str(perm_err),
                    "timestamp": now_utc,
                },
                ERROR_FIELDS,
            )
        except Exception as exc:
            errors_now += 1
            logger.warning("Failed download %s (%s): %s", doc_id, pdf_url, exc)
            append_csv_row(
                errors_csv_path,
                {
                    "doc_id": doc_id,
                    "pdf_url": pdf_url,
                    "source_url": item.get("source_url", ""),
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                    "timestamp": now_utc,
                },
                ERROR_FIELDS,
            )

    # Rewrite clean deduplicated source_register.jsonl & source_register.csv
    all_records = list(completed_by_id.values())
    write_jsonl(reg_jsonl_path, all_records)
    write_csv(reg_csv_path, all_records, SOURCE_REGISTER_FIELDS)

    logger.info(
        "Stage 2 complete: newly_downloaded=%d, errors=%d, total_registered=%d -> %s",
        downloaded_now,
        errors_now,
        len(all_records),
        reg_csv_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
