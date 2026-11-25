# ADR-001: Web Scraping, Metadata Enrichment, and Hierarchical Legal Extraction Stack

## Context
The Academic Policy Intelligence research project requires a clean, reproducible corpus of >= 300 public Universitas Gadjah Mada (UGM) policy documents (targeting 350–450 raw PDFs prior to deduplication and scanned-PDF filtering). The corpus must support:
- Ablation **A1** (fixed vs. heading-aware vs. hierarchical chunking across `BAB`, `Bagian`, `Pasal`, `Ayat`, `Huruf`)
- Ablation **A2** (BM25 / dense / hybrid / reranking + metadata filtering by `doc_type`, `year`, `status`)
- Temporal & conflict resolution (**B4** / **B6**) requiring accurate `valid_time` / `effective_date`, `status` (`Berlaku`, `Diubah`, `Dicabut`, `Unknown`), and `related_regulations`.

## Empirical Findings from Live UGM Portals
During repository/portal reconnaissance (2026-11-25):
1. `jdih.ugm.ac.id` does not resolve in DNS (`NXDOMAIN`) and `hukum.ugm.ac.id` timed out. The active official legal repository is **`https://hukor.ugm.ac.id/`** (Biro Hukum dan Organisasi UGM), which exposes Ninja Tables (`/peraturan-rektor/`, `/peraturan-mwa/`, `/daftar-keputusan-rektor/`, `/sk-2021/`..`/sk-2024/`), `/file/` pages, and structured detail pages (`/peraturan-rektor-nomor-X-tahun-YYYY/`).
2. `sa.ugm.ac.id` presents an expired TLS certificate and serves content now canonically hosted at **`https://senatakademik.ugm.ac.id/`**. `jdihn.go.id` exhibited missing intermediate CA verification in local trust stores.
3. Complementing `hukor.ugm.ac.id`, official university portals **`mwa.ugm.ac.id`**, **`senatakademik.ugm.ac.id`**, **`akademik.ugm.ac.id`**, **`pasca.ugm.ac.id`**, **`ditmawa.ugm.ac.id`**, **`penelitian.ugm.ac.id`**, **`pengabdian.ugm.ac.id`**, **`arsip.ugm.ac.id`**, **`p2l.ugm.ac.id`**, and **`ppid.ugm.ac.id`** expose open WordPress REST APIs (`wp-json/wp/v2/media?mime_type=application/pdf`) and regulation pages permitted by `robots.txt`.

## Decisions
1. **PDF Extraction (`pymupdf` >= 2.1)**: Chosen over `pypdf`/`pdfminer` for faster, layout-faithful line extraction and reliable character-density calculation (`chars/page < 200` scanned PDF filter).
2. **HTML Parsing (`beautifulsoup4` + `lxml`)**: Chosen to reliably handle Ninja Tables, relative `../wp-content/...` links, `/file/` wrapper pages, and Hukor detail metadata tables (`Tanggal berlaku`, `Status`).
3. **Controlled `.ac.id` SSL Fallback**: Strict `certifi` TLS verification is always attempted first; `verify=False` fallback is strictly limited to configured `.ac.id` / JDIH hosts experiencing expired/incomplete certificate chains, logged explicitly with warnings.
4. **No OCR on Scanned PDFs**: Per research design, image-only scanned PDFs (`< 200 chars/page`) are excluded and logged to `skipped_scanned_pdfs.csv` rather than OCR'd, preventing noisy OCR hallucinations from confounding retrieval/groundedness ablations (A1–A3).
5. **Pure Structural Markdown in `corpus_md/`**: Extracted `.md` files begin directly with `# {Title}` followed by `##` (`MENIMBANG`, `MENGINGAT`, `MEMUTUSKAN`, `BAB`, `LAMPIRAN`) and `###` (`Bagian`, `Pasal`), while structured metadata resides in `source_register.csv/.jsonl` and `corpus_clean.jsonl`. This avoids YAML frontmatter conflicts in heading-aware splitters during Ablation A1.

## Consequences
- Every stage is idempotent, resumable via JSONL checkpoints, and decoupled into `01_crawl_metadata.py`, `02_download_and_hash.py`, and `03_extract_and_filter.py`.
- Adding or overriding catalog URLs (including custom Inspect Element HTML snippets) requires only editing `sources_config.yaml` or passing `--custom-url` / `--custom-html-file`.
