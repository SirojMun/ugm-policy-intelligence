"""Shared JSONL, CSV, YAML config, and SHA-256 checkpoint utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

SOURCE_REGISTER_FIELDS = [
    "doc_id",
    "title",
    "doc_type",
    "regulation_number",
    "year",
    "valid_time",
    "effective_date",
    "status",
    "related_regulations",
    "version",
    "usage_rights",
    "source_url",
    "pdf_url",
    "access_timestamp",
    "sha256_pdf",
    "sha256_text",
    "page_count",
    "char_count",
    "chars_per_page",
    "pii_flagged",
    "pii_summary",
    "corpus_status",
    "source_name",
    "local_pdf_path",
    "markdown_path",
]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML pipeline configuration."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create all configured output, metadata, PDF, Markdown, and log directories."""
    paths = cfg["paths"]
    for key in ("output_dir", "raw_pdfs_dir", "corpus_md_dir", "metadata_dir", "logs_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    """Compute hex SHA-256 digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Compute hex SHA-256 digest for a file on disk using streaming chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute hex SHA-256 digest for normalized UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file if it exists."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single JSON record atomically to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dicts to a JSONL file atomically via temp file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(p)


def write_csv(path: str | Path, records: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write records to CSV, serializing lists/dicts cleanly as JSON strings."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for rec in records:
            for k in rec.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen

    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {}
            for k in fieldnames:
                val = rec.get(k, "")
                if isinstance(val, (list, dict)):
                    row[k] = json.dumps(val, ensure_ascii=False)
                elif val is None:
                    row[k] = ""
                else:
                    row[k] = val
            writer.writerow(row)
    tmp.replace(p)


def append_csv_row(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append one log row to a CSV file, writing header if new/empty."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        clean_row = {
            k: (json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (list, dict)) else row.get(k, ""))
            for k in fieldnames
        }
        writer.writerow(clean_row)
