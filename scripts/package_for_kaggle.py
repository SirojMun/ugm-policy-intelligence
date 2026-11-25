#!/usr/bin/env python3
"""Bundle code, clean corpus, benchmark, prompts, and red-team files into a compact zip for Kaggle upload."""

from __future__ import annotations

import zipfile
from pathlib import Path

INCLUDE_FILES = [
    "corpus_clean.jsonl",
    "source_register.csv",
    "source_register.jsonl",
    "requirements-kaggle.txt",
    "pyproject.toml",
]

INCLUDE_DIRS = [
    "src",
    "scripts",
    "prompts",
    "red_team",
    "data/benchmark",
    "data/sft",
    "docs",
]


def build_kaggle_bundle(out_zip: str | Path = "dist/ugm_policy_kaggle_bundle.zip") -> Path:
    root = Path(__file__).resolve().parent.parent
    out_path = root / out_zip
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_FILES:
            fp = root / rel
            if fp.exists():
                zf.write(fp, arcname=rel)
        for d_rel in INCLUDE_DIRS:
            dp = root / d_rel
            if not dp.exists():
                continue
            for fp in sorted(dp.rglob("*")):
                if fp.is_file() and "__pycache__" not in fp.parts:
                    zf.write(fp, arcname=str(fp.relative_to(root)))

    return out_path


if __name__ == "__main__":
    zip_path = build_kaggle_bundle()
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created Kaggle bundle: {zip_path} ({size_mb:.2f} MB)")
