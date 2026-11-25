"""Unit and integration tests for the UGM Academic Policy Intelligence scraping & extraction pipeline."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ugm_policy_pipeline.io_utils import sha256_bytes, sha256_text
from ugm_policy_pipeline.parsers import (
    extract_regulation_number,
    extract_related_regulations,
    extract_year,
    make_doc_id,
    normalize_doc_type,
    parse_hukor_detail_page,
    parse_hukor_table_page,
    parse_indonesian_date,
    should_keep_candidate,
)
from ugm_policy_pipeline.pdf_extractor import (
    extract_pdf_document,
    format_legal_markdown,
    scan_pii,
)


def _create_digital_pdf(path: Path, pages_text: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 540, 780), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def _create_blank_scanned_pdf(path: Path, n_pages: int = 2) -> None:
    doc = pymupdf.open()
    for _ in range(n_pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()


def test_parse_indonesian_date_and_reg_number():
    assert parse_indonesian_date("4 Februari 2026") == "2026-02-04"
    assert parse_indonesian_date("31 Desember 2025") == "2025-12-31"
    assert parse_indonesian_date("-") == ""
    assert extract_regulation_number("Peraturan Rektor UGM Nomor 16 Tahun 2025 tentang Tata Cara") == "16 Tahun 2025"
    assert extract_regulation_number("Keputusan Rektor Nomor 243/UN1.P/KPT/HUKOR/2024 tentang UKT") == "243/UN1.P/KPT/HUKOR/2024"
    # Amending regulation referencing an older SK/MWA number must still extract its own number & year
    amending_title = (
        "Peraturan Majelis Wali Amanat Universitas Gadjah Mada Nomor 5 Tahun 2023 tentang "
        "Perubahan Kedelapan Atas Peraturan Majelis Wali Amanat Nomor 4/SK/MWA/2014 tentang OTK"
    )
    reg_num = extract_regulation_number(amending_title)
    assert reg_num == "5 Tahun 2023"
    assert extract_year(title=amending_title, reg_num=reg_num) == 2023


def test_extract_year_prioritizes_title_over_upload_path():
    # URL upload path says 2026/04, but regulation is 2015
    yr = extract_year(
        title="Peraturan MWA UGM Nomor 4 Tahun 2015 tentang Kebijakan Umum",
        reg_num="4 Tahun 2015",
        url="https://mwa.ugm.ac.id/wp-content/uploads/sites/8/2026/04/Peraturan-MWA-UGM-Nomor-4-Tahun-2015.pdf",
    )
    assert yr == 2015


def test_parse_hukor_table_and_detail_html():
    table_html = """
    <table id="ntb_table">
      <tr><td>No</td><td>Nomor dan Judul</td><td>Kategori</td><td>Lihat dan Unduh</td></tr>
      <tr>
        <td>1</td>
        <td>Peraturan Rektor UGM Nomor 1 Tahun 2026 Tentang Fasilitas Bagi Dosen Tetap</td>
        <td>Peraturan</td>
        <td><a href="../wp-content/uploads/sites/42/2026/09/1-Tahun-2026-Fasilitas.pdf">Klik di sini</a></td>
      </tr>
    </table>
    """
    items = parse_hukor_table_page(
        table_html,
        "https://hukor.ugm.ac.id/peraturan-rektor/",
        "hukor_peraturan_rektor",
        "Peraturan Rektor",
    )
    assert len(items) == 1
    assert items[0]["pdf_url"] == "https://hukor.ugm.ac.id/wp-content/uploads/sites/42/2026/09/1-Tahun-2026-Fasilitas.pdf"
    assert items[0]["year"] == 2026
    assert items[0]["doc_type"] == "Peraturan Rektor"

    detail_html = """
    <table>
      <tr><td>Judul</td><td>Perubahan atas Peraturan Rektor Nomor 23 Tahun 2024 Tentang Pendidikan</td></tr>
      <tr><td>Nomor</td><td>Nomor 19 Tahun 2025</td></tr>
      <tr><td>Tahun</td><td>2025</td></tr>
      <tr><td>Tanggal penetapan</td><td>31 Desember 2025</td></tr>
      <tr><td>Tanggal berlaku</td><td>2 Januari 2026</td></tr>
      <tr><td>Jenis dokumen</td><td>Peraturan</td></tr>
      <tr><td>Status</td><td>Berlaku</td></tr>
      <tr><td>Unduh</td><td><a href="../wp-content/uploads/sites/42/2026/02/PR-19-2025.pdf">Klik di sini</a></td></tr>
    </table>
    """
    det = parse_hukor_detail_page(
        detail_html,
        "https://hukor.ugm.ac.id/peraturan-rektor-nomor-19-tahun-2025/",
        "hukor_detail_pages",
        "Peraturan Rektor",
    )
    assert det is not None
    assert det["effective_date"] == "2026-01-02"
    assert det["status"] == "Berlaku"
    assert det["year"] == 2025
    assert det["pdf_url"] == "https://hukor.ugm.ac.id/wp-content/uploads/sites/42/2026/02/PR-19-2025.pdf"


def test_keyword_exclusion_blocks_routine_sk_and_pii_lists():
    keep, reason = should_keep_candidate(
        title="Keputusan Rektor tentang Pembentukan Panitia Wisuda dan Penerima Beasiswa",
        url="https://hukor.ugm.ac.id/sk-panitia.pdf",
        require_policy_keyword=False,
        include_keywords=["peraturan", "pedoman"],
        exclude_keywords=["panitia", "penerima beasiswa"],
        year=2024,
        year_min=2018,
        year_max=2026,
    )
    assert not keep
    assert reason.startswith("excluded_keyword:")

    keep_ok, _ = should_keep_candidate(
        title="Peraturan Rektor UGM Nomor 23 Tahun 2024 tentang Pendidikan",
        url="https://hukor.ugm.ac.id/pr-23-2024.pdf",
        require_policy_keyword=True,
        include_keywords=["peraturan", "pedoman"],
        exclude_keywords=["panitia", "penerima beasiswa"],
        year=2024,
        year_min=2018,
        year_max=2026,
        doc_type="Peraturan Rektor",
    )
    assert keep_ok

    keep_normative_pr, _ = should_keep_candidate(
        title="Peraturan Rektor Nomor 16 Tahun 2024 Tentang Penugasan Direksi di BUMN",
        url="https://hukor.ugm.ac.id/pr-16-2024.pdf",
        require_policy_keyword=False,
        include_keywords=["peraturan"],
        exclude_keywords=["penugasan", "panitia"],
        year=2024,
        year_min=2018,
        year_max=2026,
        doc_type="Peraturan Rektor",
    )
    assert keep_normative_pr

    skip_sk, _ = should_keep_candidate(
        title="Keputusan Rektor tentang Penugasan Personal Tim Kerja",
        url="https://hukor.ugm.ac.id/sk-penugasan.pdf",
        require_policy_keyword=False,
        include_keywords=["keputusan"],
        exclude_keywords=["penugasan", "panitia"],
        year=2024,
        year_min=2018,
        year_max=2026,
        doc_type="Keputusan Rektor",
    )
    assert not skip_sk


def test_format_legal_markdown_hierarchy_and_header_footer_cleaning():
    pages = [
        "-1-\nHEADER KOP UGM\nREKTOR UNIVERSITAS GADJAH MADA\nMenimbang : a. bahwa untuk tertib\nadministrasi akademik;\nMengingat : 1. Peraturan Pemerintah Nomor 67 Tahun 2013 tentang Statuta UGM;\nMEMUTUSKAN:\nMenetapkan : PERATURAN REKTOR TENTANG PENDIDIKAN.\nFOOTER RESMI UGM",
        "-2-\nHEADER KOP UGM\nBAB I\nKETENTUAN UMUM\nBagian Kesatu\nRuang Lingkup\nPasal 1\n(1)\nUniversitas adalah\nUniversitas Gadjah Mada sebagaimana dimaksud pada ayat\n(2) di bawah ini.\na.\nsarjana; dan\nb.\npascasarjana.\nFOOTER RESMI UGM",
        "-3-\nHEADER KOP UGM\nBAB II\nPENUTUP\nPasal 2\nPada saat peraturan ini berlaku, Peraturan Rektor Nomor 13 Tahun 2022 dicabut dan dinyatakan tidak berlaku.\nLAMPIRAN I\nRincian Kurikulum\nFOOTER RESMI UGM",
    ]
    md = format_legal_markdown("Peraturan Rektor Nomor 23 Tahun 2024 tentang Pendidikan", pages)
    assert md.startswith("# Peraturan Rektor Nomor 23 Tahun 2024 tentang Pendidikan")
    assert "## MENIMBANG" in md
    assert "## MENGINGAT" in md
    assert "## MEMUTUSKAN" in md
    assert "## BAB I — KETENTUAN UMUM" in md
    assert "### Bagian Kesatu — Ruang Lingkup" in md
    assert "### Pasal 1" in md
    assert "(1) Universitas adalah Universitas Gadjah Mada sebagaimana dimaksud pada ayat (2) di bawah ini." in md
    assert "a. sarjana; dan" in md
    assert "## LAMPIRAN I" in md
    assert "HEADER KOP UGM" not in md
    assert "FOOTER RESMI UGM" not in md
    assert "-1-" not in md


def test_scanned_pdf_detection_and_digital_extraction(tmp_path: Path):
    scanned = tmp_path / "scanned.pdf"
    _create_blank_scanned_pdf(scanned, n_pages=3)
    res_scan = extract_pdf_document(scanned, "Dokumen Scan", min_chars_per_page=200)
    assert res_scan.is_scanned is True
    assert res_scan.chars_per_page < 200

    digital = tmp_path / "digital.pdf"
    long_body = (
        "Menimbang : a. bahwa pendidikan tinggi merupakan bagian integral dalam pembangunan nasional "
        "dan penyelenggaraan tridharma perguruan tinggi di Universitas Gadjah Mada secara berkelanjutan.\n"
        "BAB I\nKETENTUAN UMUM\nPasal 1\n(1) Pedoman akademik ini mengatur kurikulum, evaluasi hasil belajar, "
        "beban studi, masa studi, serta hak dan kewajiban mahasiswa program sarjana dan pascasarjana.\n"
        "Ditetapkan di Yogyakarta pada tanggal 10 Januari 2025\n"
    )
    _create_digital_pdf(digital, [long_body, long_body])
    res_dig = extract_pdf_document(digital, "Peraturan Rektor UGM Nomor 5 Tahun 2025", min_chars_per_page=200)
    assert res_dig.is_scanned is False
    assert res_dig.effective_date == "2025-01-10"
    assert "## BAB I — KETENTUAN UMUM" in res_dig.markdown_text
    assert "### Pasal 1" in res_dig.markdown_text


def test_pii_scanner_flags_mass_nik_nip_nim_bank():
    clean_text = "Pasal 1 Peraturan Rektor Nomor 16 Tahun 2025 berlaku bagi sivitas akademika UGM."
    flagged_clean, _ = scan_pii(clean_text)
    assert not flagged_clean

    pii_text = "\n".join(
        [
            "NIK 3404011205990001, 3404011205990002, 3404011205990003",
            "NIM 21/473100/TK/52121 21/473101/TK/52122 21/473102/TK/52123 21/473103/TK/52124 21/473104/TK/52125",
            "No. Rekening: 1234567890123 Bank Mandiri, No. Rek: 9876543210987",
        ]
    )
    flagged, summary = scan_pii(pii_text)
    assert flagged
    assert summary["nik_count"] == 3
    assert summary["nim_count"] == 5
    assert summary["bank_account_count"] == 2


def test_doc_id_and_hash_dedup_helpers():
    assert make_doc_id("Peraturan Rektor", 2023, "15 Tahun 2023", 1) == "UGM_PR_2023_015"
    assert make_doc_id("Peraturan MWA", 2025, "1 Tahun 2025", 2) == "UGM_MWA_2025_001"
    assert sha256_bytes(b"abc") == sha256_text("abc")
    related = extract_related_regulations("Mengingat Peraturan Rektor Universitas Gadjah Mada Nomor 23 Tahun 2024 tentang Pendidikan")
    assert any("23 Tahun 2024" in r for r in related)
