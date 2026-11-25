#!/usr/bin/env python3
"""Stage 1: Ethical metadata & catalog crawler for public UGM regulation portals."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ugm_policy_pipeline.http_client import EthicalHttpClient, HttpConfig
from ugm_policy_pipeline.io_utils import (
    append_jsonl,
    ensure_dirs,
    load_config,
    load_jsonl,
    write_csv,
    write_jsonl,
)
from ugm_policy_pipeline.parsers import (
    make_doc_id,
    parse_html_pdf_links,
    parse_hukor_detail_page,
    parse_hukor_table_page,
    parse_wp_media_items,
    should_keep_candidate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("01_crawl_metadata")


def canonical_pdf_key(url: str) -> str:
    """Normalize HTTP/HTTPS and trailing slashes for PDF URL deduplication."""
    parsed = urlparse(url.strip())
    return f"{parsed.netloc.lower()}{parsed.path}"


def resolve_wrapper_pdf(client: EthicalHttpClient, wrapper_url: str, cache: dict[str, str]) -> str:
    """Follow a /file/ or /download/ wrapper page to discover its embedded .pdf link."""
    if wrapper_url in cache:
        return cache[wrapper_url]
    try:
        resp = client.get(wrapper_url)
        if urlparse(resp.url).path.lower().endswith(".pdf"):
            cache[wrapper_url] = resp.url
            return resp.url
        parsed = parse_html_pdf_links(resp.text, resp.url, "wrapper", "Peraturan Rektor")
        resolved = parsed[0]["pdf_url"] if parsed else ""
        cache[wrapper_url] = resolved
        return resolved
    except Exception as exc:
        logger.warning("Could not resolve wrapper %s: %s", wrapper_url, exc)
        cache[wrapper_url] = ""
        return ""


def crawl_hukor_wp_pages(
    client: EthicalHttpClient,
    src: dict[str, Any],
    crawled_urls: set[str],
) -> list[dict[str, Any]]:
    """Discover and parse individual Hukor regulation detail pages and /file/ sitemap pages."""
    items: list[dict[str, Any]] = []
    api_url = src["api_url"]
    source_name = src["name"]
    default_doc_type = src.get("default_doc_type", "Peraturan Rektor")

    # Fetch WP pages index via REST API
    page = 1
    candidate_pages: list[str] = []
    while True:
        paged_api = f"{api_url}?per_page=100&page={page}"
        try:
            resp = client.get(paged_api)
            batch = resp.json()
        except Exception as exc:
            logger.info("WP pages pagination stopped at page %d (%s)", page, exc)
            break
        if not batch:
            break
        for p in batch:
            link = (p.get("link") or "").strip()
            if re.search(r"peraturan-(?:rektor|mwa|senat)-nomor-", link, re.IGNORECASE):
                candidate_pages.append(link)
        total_pages = int(resp.headers.get("X-WP-TotalPages") or resp.headers.get("x-wp-totalpages") or 1)
        if page >= total_pages:
            break
        page += 1

    # Also check /wp-sitemap-posts-file-1.xml if configured
    sitemap_url = src.get("sitemap_file_url")
    if sitemap_url:
        try:
            s_resp = client.get(sitemap_url)
            root = ET.fromstring(s_resp.text)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for loc in root.findall(".//sm:loc", ns):
                if loc.text:
                    candidate_pages.append(loc.text.strip())
        except Exception as exc:
            logger.warning("Failed fetching sitemap %s: %s", sitemap_url, exc)

    seen_local: set[str] = set()
    for url in candidate_pages:
        if url in seen_local or url in crawled_urls:
            continue
        seen_local.add(url)
        try:
            resp = client.get(url)
            rec = parse_hukor_detail_page(resp.text, resp.url, source_name, default_doc_type)
            if rec:
                rec["priority"] = src.get("priority", 1)
                rec["require_policy_keyword"] = src.get("require_policy_keyword", False)
                items.append(rec)
        except Exception as exc:
            logger.warning("Error parsing detail page %s: %s", url, exc)

    return items


def crawl_wp_media(client: EthicalHttpClient, src: dict[str, Any]) -> list[dict[str, Any]]:
    """Crawl WordPress REST API PDF attachments cleanly."""
    items: list[dict[str, Any]] = []
    base = src["base_url"].rstrip("/")
    max_pages = int(src.get("max_pages", 6))
    source_name = src["name"]
    default_doc_type = src.get("default_doc_type", "Pedoman Layanan")

    for page in range(1, max_pages + 1):
        api_url = f"{base}/wp-json/wp/v2/media?mime_type=application/pdf&per_page=100&page={page}"
        try:
            resp = client.get(api_url)
            batch = resp.json()
        except Exception as exc:
            logger.info("Stopped WP media crawl for %s at page %d (%s)", base, page, exc)
            break
        if not batch:
            break
        parsed = parse_wp_media_items(batch, source_name=source_name, default_doc_type=default_doc_type)
        for rec in parsed:
            rec["priority"] = src.get("priority", 2)
            rec["require_policy_keyword"] = src.get("require_policy_keyword", True)
        items.extend(parsed)
        total_pages = int(resp.headers.get("X-WP-TotalPages") or resp.headers.get("x-wp-totalpages") or 1)
        if page >= total_pages:
            break
    return items


def merge_and_filter_catalog(
    raw_items: list[dict[str, Any]],
    cfg: dict[str, Any],
    year_min: int,
    year_max: int,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge duplicate PDF entries (preserving richest metadata) and apply policy/year filters."""
    filt_cfg = cfg["filtering"]
    include_kw = filt_cfg["policy_include_keywords"]
    exclude_kw = filt_cfg["exclude_keywords"]
    usage_rights = cfg["project"]["usage_rights"]

    # Merge by canonical PDF URL
    merged_by_pdf: dict[str, dict[str, Any]] = {}
    for rec in raw_items:
        pdf_url = (rec.get("pdf_url") or "").strip()
        if not pdf_url:
            continue
        key = canonical_pdf_key(pdf_url)
        existing = merged_by_pdf.get(key)
        if existing is None:
            merged_by_pdf[key] = dict(rec)
        else:
            # Prefer HTTPS, lower priority number, richer effective_date/status/reg_num
            if pdf_url.startswith("https://") and not existing["pdf_url"].startswith("https://"):
                existing["pdf_url"] = pdf_url
            if rec.get("priority", 9) < existing.get("priority", 9):
                existing["priority"] = rec["priority"]
                existing["doc_type"] = rec.get("doc_type") or existing["doc_type"]
                existing["source_name"] = rec.get("source_name") or existing["source_name"]
            if not existing.get("effective_date") and rec.get("effective_date"):
                existing["effective_date"] = rec["effective_date"]
                existing["valid_time"] = rec.get("valid_time") or rec["effective_date"]
            if existing.get("status", "Unknown") == "Unknown" and rec.get("status", "Unknown") != "Unknown":
                existing["status"] = rec["status"]
            if not existing.get("regulation_number") and rec.get("regulation_number"):
                existing["regulation_number"] = rec["regulation_number"]
            if not existing.get("year") and rec.get("year"):
                existing["year"] = rec["year"]
            if len(rec.get("title", "")) > len(existing.get("title", "")) and "tentang" in rec.get("title", "").lower():
                existing["title"] = rec["title"]
            # Prefer individual detail page as source_url over bulk table URL
            if "peraturan-" in (rec.get("source_url") or "") and "nomor-" in (rec.get("source_url") or ""):
                existing["source_url"] = rec["source_url"]
            # If any source didn't require policy keyword (e.g., curated Hukor table), keep False
            existing["require_policy_keyword"] = bool(
                existing.get("require_policy_keyword", False) and rec.get("require_policy_keyword", False)
            )

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    # Sort deterministically by priority ascending, then year descending, then title
    candidates = sorted(
        merged_by_pdf.values(),
        key=lambda r: (int(r.get("priority", 9)), -(r.get("year") or 0), r.get("title", "")),
    )

    used_ids: dict[str, int] = {}
    for idx, rec in enumerate(candidates, start=1):
        ok, reason = should_keep_candidate(
            title=rec.get("title", ""),
            url=rec.get("pdf_url", ""),
            require_policy_keyword=bool(rec.get("require_policy_keyword", False)),
            include_keywords=include_kw,
            exclude_keywords=exclude_kw,
            year=rec.get("year"),
            year_min=year_min,
            year_max=year_max,
            doc_type=rec.get("doc_type", ""),
        )
        if not ok:
            skipped.append(
                {
                    "title": rec.get("title", ""),
                    "pdf_url": rec.get("pdf_url", ""),
                    "source_name": rec.get("source_name", ""),
                    "year": rec.get("year") or "",
                    "reason": reason,
                }
            )
            continue

        if len(kept) >= max_candidates:
            skipped.append(
                {
                    "title": rec.get("title", ""),
                    "pdf_url": rec.get("pdf_url", ""),
                    "source_name": rec.get("source_name", ""),
                    "year": rec.get("year") or "",
                    "reason": f"max_candidates_cap:{max_candidates}",
                }
            )
            continue

        base_id = make_doc_id(rec["doc_type"], rec.get("year"), rec.get("regulation_number", ""), seq=len(kept) + 1)
        count = used_ids.get(base_id, 0) + 1
        used_ids[base_id] = count
        doc_id = base_id if count == 1 else f"{base_id}_V{count}"

        kept.append(
            {
                "doc_id": doc_id,
                "title": rec["title"],
                "doc_type": rec["doc_type"],
                "regulation_number": rec.get("regulation_number", ""),
                "year": rec.get("year"),
                "valid_time": rec.get("valid_time") or rec.get("effective_date") or (str(rec["year"]) if rec.get("year") else ""),
                "effective_date": rec.get("effective_date", ""),
                "status": rec.get("status", "Unknown"),
                "related_regulations": rec.get("related_regulations", []),
                "version": rec.get("version", "original"),
                "usage_rights": usage_rights,
                "source_url": rec.get("source_url", ""),
                "pdf_url": rec["pdf_url"],
                "source_name": rec.get("source_name", ""),
                "priority": rec.get("priority", 1),
            }
        )

    return kept, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: Crawl UGM public regulation metadata & PDF links.")
    parser.add_argument("--config", default="sources_config.yaml", help="Path to YAML config file.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Override max filtered candidates (default from config).")
    parser.add_argument("--min-year", type=int, default=None, help="Override min publication year (default 2018).")
    parser.add_argument("--max-year", type=int, default=None, help="Override max publication year (default 2026).")
    parser.add_argument("--only-sources", nargs="*", default=None, help="Optional list of source names from config to crawl.")
    parser.add_argument("--custom-url", default=None, help="Optional extra catalog URL to crawl in addition to config.")
    parser.add_argument("--custom-html-file", default=None, help="Optional local HTML snippet file (from Inspect Element) to parse.")
    parser.add_argument("--custom-base-url", default="https://hukor.ugm.ac.id/", help="Base URL used to resolve relative links in --custom-html-file.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing catalog_raw.jsonl checkpoint and recrawl.")
    parser.add_argument("--no-delay", action="store_true", help="Disable rate-limit sleep (only for offline/mock testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    paths = cfg["paths"]
    http_cfg = cfg["http"]

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

    raw_path = Path(paths["catalog_raw"])
    existing_raw = [] if args.no_resume else load_jsonl(raw_path)
    if args.no_resume and raw_path.exists():
        raw_path.unlink()

    crawled_sources = {r.get("_crawled_marker") for r in existing_raw if r.get("_crawled_marker")}
    raw_items = [r for r in existing_raw if not r.get("_crawled_marker")]
    wrapper_cache: dict[str, str] = {}

    for src in cfg["sources"]:
        name = src["name"]
        if args.only_sources and name not in args.only_sources:
            continue
        if name in crawled_sources:
            logger.info("[checkpoint] Skipping already crawled source: %s", name)
            continue

        logger.info("Crawling source: %s (type=%s)", name, src["type"])
        new_items: list[dict[str, Any]] = []
        stype = src["type"]

        if stype == "hukor_table_page":
            urls = src.get("urls") or [src["url"]]
            for u in urls:
                try:
                    resp = client.get(u)
                    parsed = parse_hukor_table_page(resp.text, resp.url, name, src["default_doc_type"])
                    for item in parsed:
                        if not item.get("pdf_url") and item.get("wrapper_url"):
                            item["pdf_url"] = resolve_wrapper_pdf(client, item["wrapper_url"], wrapper_cache)
                        item["priority"] = src.get("priority", 1)
                        item["require_policy_keyword"] = src.get("require_policy_keyword", False)
                        if item.get("pdf_url"):
                            new_items.append(item)
                except Exception as exc:
                    logger.warning("Failed table page %s: %s", u, exc)

        elif stype == "hukor_wp_pages":
            crawled_detail_urls = {r.get("source_url") for r in raw_items if r.get("source_name") == name}
            new_items = crawl_hukor_wp_pages(client, src, crawled_detail_urls)

        elif stype == "html_pdf_page":
            urls = src.get("urls") or [src["url"]]
            for u in urls:
                try:
                    resp = client.get(u)
                    parsed = parse_html_pdf_links(resp.text, resp.url, name, src["default_doc_type"])
                    for item in parsed:
                        item["priority"] = src.get("priority", 2)
                        item["require_policy_keyword"] = src.get("require_policy_keyword", False)
                    new_items.extend(parsed)
                except Exception as exc:
                    logger.warning("Failed html_pdf_page %s: %s", u, exc)

        elif stype == "wp_media":
            new_items = crawl_wp_media(client, src)

        else:
            logger.warning("Unknown source type '%s' for %s", stype, name)

        for item in new_items:
            append_jsonl(raw_path, item)
        append_jsonl(raw_path, {"_crawled_marker": name})
        raw_items.extend(new_items)
        logger.info("Source %s yielded %d candidate items.", name, len(new_items))

    # Optional user-supplied custom URL or Inspect Element HTML file
    if args.custom_url:
        logger.info("Crawling custom catalog URL: %s", args.custom_url)
        try:
            resp = client.get(args.custom_url)
            parsed = parse_hukor_table_page(resp.text, resp.url, "custom_url", "Peraturan Rektor")
            if not parsed:
                parsed = parse_html_pdf_links(resp.text, resp.url, "custom_url", "Peraturan Rektor")
            for item in parsed:
                if not item.get("pdf_url") and item.get("wrapper_url"):
                    item["pdf_url"] = resolve_wrapper_pdf(client, item["wrapper_url"], wrapper_cache)
                item["priority"] = 1
                item["require_policy_keyword"] = False
                if item.get("pdf_url"):
                    raw_items.append(item)
        except Exception as exc:
            logger.warning("Failed custom URL %s: %s", args.custom_url, exc)

    if args.custom_html_file:
        html_text = Path(args.custom_html_file).read_text(encoding="utf-8")
        parsed = parse_hukor_table_page(html_text, args.custom_base_url, "custom_html", "Peraturan Rektor")
        if not parsed:
            parsed = parse_html_pdf_links(html_text, args.custom_base_url, "custom_html", "Peraturan Rektor")
        for item in parsed:
            item["priority"] = 1
            item["require_policy_keyword"] = False
            if item.get("pdf_url"):
                raw_items.append(item)
        logger.info("Parsed %d items from custom HTML file %s", len(parsed), args.custom_html_file)

    year_min = args.min_year if args.min_year is not None else int(cfg["project"]["target_year_min"])
    year_max = args.max_year if args.max_year is not None else int(cfg["project"]["target_year_max"])
    max_candidates = args.max_candidates if args.max_candidates is not None else int(cfg["project"]["target_max_candidates"])

    kept, skipped = merge_and_filter_catalog(
        raw_items=raw_items,
        cfg=cfg,
        year_min=year_min,
        year_max=year_max,
        max_candidates=max_candidates,
    )

    write_jsonl(paths["catalog_filtered"], kept)
    write_csv(paths["skipped_filter_csv"], skipped, ["title", "pdf_url", "source_name", "year", "reason"])

    logger.info(
        "Stage 1 complete: raw=%d, filtered_kept=%d, skipped=%d -> %s",
        len(raw_items),
        len(kept),
        len(skipped),
        paths["catalog_filtered"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
