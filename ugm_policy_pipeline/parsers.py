"""Catalog parsers, Indonesian regulation metadata extractors, and policy filtering."""

from __future__ import annotations

import html as html_lib
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

MONTH_MAP = {
    "januari": 1,
    "jan": 1,
    "februari": 2,
    "feb": 2,
    "pebruari": 2,
    "maret": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "mei": 5,
    "may": 5,
    "juni": 6,
    "jun": 6,
    "juli": 7,
    "jul": 7,
    "agustus": 8,
    "agu": 8,
    "ags": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "oktober": 10,
    "okt": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "nopember": 11,
    "desember": 12,
    "des": 12,
    "dec": 12,
}

DOC_TYPE_PREFIX = {
    "Peraturan Rektor": "PR",
    "Peraturan MWA": "MWA",
    "Peraturan Senat Akademik": "SA",
    "Keputusan Rektor": "KR",
    "Keputusan MWA": "KMWA",
    "Keputusan Senat Akademik": "KSA",
    "Pedoman Akademik": "PED_AKD",
    "Pedoman Kemahasiswaan": "PED_MHS",
    "Pedoman Layanan": "PED_LYN",
    "Statuta / Eksternal": "EXT",
}


def clean_whitespace(text: str) -> str:
    """Decode HTML entities and normalize whitespace."""
    unescaped = html_lib.unescape(text or "")
    return re.sub(r"\s+", " ", unescaped).strip()


def parse_indonesian_date(raw: str) -> str:
    """Convert Indonesian or ISO date strings (e.g. '4 Februari 2026') to YYYY-MM-DD."""
    s = clean_whitespace(raw)
    if not s or s in ("-", "—", "N/A"):
        return ""

    iso_match = re.search(r"\b(20\d{2}|19\d{2})-(\d{2})-(\d{2})\b", s)
    if iso_match:
        return iso_match.group(0)

    dmy_slash = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})\b", s)
    if dmy_slash:
        day, mon, yr = int(dmy_slash.group(1)), int(dmy_slash.group(2)), int(dmy_slash.group(3))
        if 1 <= mon <= 12 and 1 <= day <= 31:
            return f"{yr:04d}-{mon:02d}-{day:02d}"

    id_match = re.search(
        r"\b(\d{1,2})\s+([A-Za-z]+)\s+((?:19|20)\d{2})\b",
        s,
        flags=re.IGNORECASE,
    )
    if id_match:
        day = int(id_match.group(1))
        mon = MONTH_MAP.get(id_match.group(2).lower())
        yr = int(id_match.group(3))
        if mon and 1 <= day <= 31:
            return f"{yr:04d}-{mon:02d}-{day:02d}"
    return ""


def normalize_doc_type(title: str, category: str = "", url: str = "", default: str = "Pedoman Layanan") -> str:
    """Infer standardized UGM regulation category from title, table category, and URL."""
    combined = f"{title} {category} {url}".lower()
    if "peraturan rektor" in combined or re.search(r"\bpr[-_/ ]\d+", combined):
        return "Peraturan Rektor"
    if "peraturan majelis wali amanat" in combined or "peraturan mwa" in combined or "permwa" in combined or "per-mwa" in combined or "per_mwa" in combined:
        return "Peraturan MWA"
    if "peraturan senat" in combined or "senat akademik" in combined and "peraturan" in combined:
        return "Peraturan Senat Akademik"
    if "keputusan senat" in combined or "sk-sa" in combined or "sk sa" in combined or "keputusan-sa" in combined or "tata tertib sa" in combined or "tata-tertib-sa" in combined:
        return "Keputusan Senat Akademik"
    if "keputusan majelis wali amanat" in combined or "keputusan mwa" in combined or "sk-mwa" in combined or "sk mwa" in combined:
        return "Keputusan MWA"
    if "keputusan rektor" in combined or "keptor" in combined or "/kpt/" in combined or "sk rektor" in combined or "sk-rektor" in combined or re.search(r"\bsk[-_ ]\d+", combined):
        return "Keputusan Rektor"
    if any(k in combined for k in ("peraturan pemerintah", "undang-undang", "permen", "statuta")):
        return "Statuta / Eksternal"
    if any(k in combined for k in ("pedoman akademik", "panduan akademik", "kalender akademik", "kurikulum", "aturan akademik")):
        return "Pedoman Akademik"
    if "kemahasiswaan" in combined or "ditmawa" in url.lower() or "siwira" in combined:
        return "Pedoman Kemahasiswaan"
    if any(k in combined for k in ("pedoman", "panduan", "petunjuk", "juknis", "standar", "kebijakan", "sop", "pos-")):
        return default if default.startswith("Pedoman") else "Pedoman Layanan"
    return default


def extract_regulation_number(text: str) -> str:
    """Extract formal regulation number (e.g. '16 Tahun 2025', '243/UN1.P/KPT/HUKOR/2024')."""
    s = clean_whitespace(text)
    # Inspect the subject's own header (before 'tentang') first to avoid picking referenced numbers
    head = re.split(r"\btentang\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    patterns = [
        r"(?:Nomor|No\.?)\s*([0-9A-Za-z./-]+/UN1[^,\s;)]*/(?:19|20)\d{2})",
        r"(?:Nomor|No\.?)\s*(\d+/SK/MWA/(?:19|20)\d{2})",
        r"(?:Nomor|No\.?)\s*(\d+\s+Tahun\s+(?:19|20)\d{2})",
        r"(?:Nomor|No\.?)\s*(\d+/\d{4})",
        r"\b(?:PR|PerMWA|Per-MWA|SK-SA|SK-MWA|Keputusan-SA)[-_ ]?(?:No(?:mor)?[-_ ]?)?(\d+[-_ ]?(?:Tahun|Th)?[-_ ]?(?:19|20)\d{2})\b",
        r"\b(\d{1,4}[-_/ ](?:19|20)\d{2})\b",
    ]
    for region in (head, s):
        best_pos: int | None = None
        best_val = ""
        for pat in patterns:
            m = re.search(pat, region, flags=re.IGNORECASE)
            if m and (best_pos is None or m.start() < best_pos):
                best_pos = m.start()
                best_val = clean_whitespace(m.group(1)).replace("_", "-")
        if best_val:
            return best_val
    return ""


def extract_year(
    title: str = "",
    reg_num: str = "",
    explicit_year: str | int | None = None,
    effective_date: str = "",
    url: str = "",
    wp_date: str = "",
) -> int | None:
    """Extract publication/enactment year prioritizing explicit metadata & own title prefix over upload path."""
    if explicit_year:
        m = re.search(r"\b((?:19|20)\d{2})\b", str(explicit_year))
        if m:
            return int(m.group(1))

    title_head = re.split(r"\btentang\b", title or "", maxsplit=1, flags=re.IGNORECASE)[0]
    for candidate in (reg_num, title_head, title):
        if not candidate:
            continue
        m_th = re.search(r"(?:Tahun|Th\.?|/)\s*((?:19|20)\d{2})\b", candidate, flags=re.IGNORECASE)
        if m_th:
            return int(m_th.group(1))
        years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", candidate)]
        if years:
            return years[0] if candidate is title_head else years[-1]

    if effective_date and re.match(r"^(19|20)\d{2}", effective_date):
        return int(effective_date[:4])

    # Check filename before upload folder YYYY/MM
    filename = urlparse(url).path.rsplit("/", 1)[-1]
    fn_years = [int(y) for y in re.findall(r"(?:^|[^0-9])((?:19|20)\d{2})(?:[^0-9]|$)", filename)]
    if fn_years:
        return fn_years[0]

    # Fallback: WP publish date or /YYYY/MM/ upload path
    if wp_date and re.match(r"^(19|20)\d{2}", wp_date):
        return int(wp_date[:4])
    path_m = re.search(r"/((?:19|20)\d{2})/\d{2}/", url)
    if path_m:
        return int(path_m.group(1))
    return None


def extract_version(title: str, url: str = "") -> str:
    """Detect amendment/revision version markers from title or filename."""
    s = f"{title} {url}".lower()
    if "perubahan keempat" in s or "revisi 4" in s:
        return "perubahan-4"
    if "perubahan ketiga" in s or "revisi 3" in s:
        return "perubahan-3"
    if "perubahan kedua" in s or "revisi 2" in s:
        return "perubahan-2"
    if "perubahan atas" in s or "perubahan pertama" in s or "revisi" in s:
        m = re.search(r"revisi(?:\s+tahun)?\s*((?:19|20)\d{2}|\d+)", s)
        return f"revisi-{m.group(1)}" if m else "perubahan-1"
    return "original"


def normalize_status(raw: str, title: str = "") -> str:
    """Normalize regulation status into {Berlaku, Diubah, Dicabut, Unknown}."""
    s = clean_whitespace(raw).lower()
    if "dicabut" in s or "tidak berlaku" in s:
        return "Dicabut"
    if "diubah" in s:
        return "Diubah"
    if "berlaku" in s:
        return "Berlaku"
    if "perubahan" in title.lower():
        # Note: an amending regulation itself is Berlaku unless superseded, keep Unknown if unverified
        return "Unknown"
    return "Unknown"


def extract_related_regulations(text: str) -> list[str]:
    """Extract referenced regulation identifiers from title, Mengingat, or body text."""
    s = clean_whitespace(text)
    patterns = [
        r"(?:Peraturan\s+(?:Rektor|Majelis\s+Wali\s+Amanat|MWA|Senat\s+Akademik|Pemerintah|Menteri[^,.;]*)\s+(?:Universitas\s+Gadjah\s+Mada\s+)?Nomor\s+[0-9A-Za-z./-]+\s+Tahun\s+(?:19|20)\d{2})",
        r"(?:Keputusan\s+(?:Rektor|Majelis\s+Wali\s+Amanat|MWA|Senat\s+Akademik)\s+(?:Universitas\s+Gadjah\s+Mada\s+)?Nomor\s+[0-9A-Za-z./-]+\s*(?:Tahun\s+(?:19|20)\d{2})?)",
        r"(?:Undang-Undang\s+Nomor\s+\d+\s+Tahun\s+(?:19|20)\d{2})",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            val = clean_whitespace(m.group(0))
            key = val.lower()
            if key not in seen:
                seen.add(key)
                found.append(val)
    return found


def make_doc_id(doc_type: str, year: int | None, reg_num: str, seq: int) -> str:
    """Build deterministic standardized doc_id such as UGM_PR_2023_015."""
    prefix = DOC_TYPE_PREFIX.get(doc_type, "DOC")
    yr_str = str(year) if year else "0000"
    num_match = re.match(r"(\d+)", reg_num or "")
    if num_match:
        num_part = f"{int(num_match.group(1)):03d}"
    else:
        num_part = f"X{seq:03d}"
    return f"UGM_{prefix}_{yr_str}_{num_part}"


def should_keep_candidate(
    title: str,
    url: str,
    require_policy_keyword: bool,
    include_keywords: list[str],
    exclude_keywords: list[str],
    year: int | None,
    year_min: int,
    year_max: int,
    doc_type: str = "",
) -> tuple[bool, str]:
    """Filter candidate by routine/PII exclusions, policy keywords, and year window."""
    combined = f"{title} {url}".lower()

    # Always block drafts/testing and explicit PII-risk lists even on Peraturan pages
    always_block = ("draf ", "draf-", "draft ", "draft-", "rancangan ", "tetsing", "testing", "daftar penerima", "penerima beasiswa")
    for bad in always_block:
        if bad in combined:
            return False, f"excluded_keyword:{bad}"

    # Normative regulations (Peraturan Rektor/MWA/SA) are general policies, not individual SKs
    is_normative_peraturan = doc_type in ("Peraturan Rektor", "Peraturan MWA", "Peraturan Senat Akademik")
    if not is_normative_peraturan:
        for bad in exclude_keywords:
            if bad.lower() in combined:
                return False, f"excluded_keyword:{bad}"

    if require_policy_keyword:
        if not any(kw.lower() in combined for kw in include_keywords):
            return False, "missing_policy_keyword"

    if year is not None and (year < year_min or year > year_max):
        return False, f"out_of_year_range:{year}"

    return True, "ok"


def parse_hukor_table_page(html: str, page_url: str, source_name: str, default_doc_type: str) -> list[dict[str, Any]]:
    """Parse Ninja Tables / standard HTML tables on hukor.ugm.ac.id catalog pages."""
    soup = BeautifulSoup(html, "lxml")
    items: list[dict[str, Any]] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [clean_whitespace(c.get_text(" ", strip=True)).lower() for c in rows[0].find_all(["th", "td"])]
        if not any("judul" in h or "nomor" in h for h in headers):
            continue

        for tr in rows[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            cell_texts = [clean_whitespace(c.get_text(" ", strip=True)) for c in cells]

            # Map columns flexibly based on header names
            title = ""
            category = ""
            explicit_year = ""
            for idx, h in enumerate(headers):
                if idx >= len(cell_texts):
                    break
                if "judul" in h or "nama" in h:
                    title = cell_texts[idx]
                elif "kategori" in h or "jenis" in h:
                    category = cell_texts[idx]
                elif "tahun" in h:
                    explicit_year = cell_texts[idx]

            if not title and len(cell_texts) >= 2:
                title = max(cell_texts, key=len)

            links: list[str] = []
            for a in tr.find_all("a", href=True):
                href = urljoin(page_url, a["href"].strip())
                if href not in links:
                    links.append(href)

            if not title or not links:
                continue

            pdf_links = [u for u in links if urlparse(u).path.lower().endswith(".pdf")]
            wrapper_links = [u for u in links if u not in pdf_links]
            primary_link = pdf_links[0] if pdf_links else wrapper_links[0]
            reg_num = extract_regulation_number(title)
            doc_type = normalize_doc_type(title, category=category, url=primary_link, default=default_doc_type)
            year = extract_year(title=title, reg_num=reg_num, explicit_year=explicit_year, url=primary_link)

            items.append(
                {
                    "title": title,
                    "doc_type": doc_type,
                    "regulation_number": reg_num,
                    "year": year,
                    "valid_time": "",
                    "effective_date": "",
                    "status": "Unknown",
                    "related_regulations": extract_related_regulations(title),
                    "version": extract_version(title, primary_link),
                    "source_url": page_url if pdf_links else primary_link,
                    "pdf_url": pdf_links[0] if pdf_links else "",
                    "wrapper_url": wrapper_links[0] if not pdf_links and wrapper_links else "",
                    "source_name": source_name,
                }
            )
    return items


def parse_hukor_detail_page(html: str, page_url: str, source_name: str, default_doc_type: str) -> dict[str, Any] | None:
    """Parse individual Hukor regulation detail page metadata table (Status, Tanggal berlaku, etc.)."""
    soup = BeautifulSoup(html, "lxml")
    meta: dict[str, str] = {}
    pdf_url = ""

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) >= 2:
                key = clean_whitespace(cells[0].get_text(" ", strip=True)).lower()
                val = clean_whitespace(cells[1].get_text(" ", strip=True))
                if key:
                    meta[key] = val
                for a in cells[1].find_all("a", href=True):
                    href = urljoin(page_url, a["href"].strip())
                    if urlparse(href).path.lower().endswith(".pdf"):
                        pdf_url = href

    if not pdf_url:
        for a in soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"].strip())
            if urlparse(href).path.lower().endswith(".pdf"):
                pdf_url = href
                break

    title = meta.get("judul") or ""
    if not title:
        h1 = soup.find(["h1", "h2"])
        title = clean_whitespace(h1.get_text(" ", strip=True)) if h1 else ""

    if not title or not pdf_url:
        return None

    reg_num = meta.get("nomor") or extract_regulation_number(title)
    doc_type = normalize_doc_type(
        f"{meta.get('jenis dokumen', '')} {title}",
        category=meta.get("jenis dokumen", ""),
        url=pdf_url,
        default=default_doc_type,
    )
    eff_date = parse_indonesian_date(meta.get("tanggal berlaku", "")) or parse_indonesian_date(meta.get("tanggal penetapan", ""))
    year = extract_year(
        title=title,
        reg_num=reg_num,
        explicit_year=meta.get("tahun"),
        effective_date=eff_date,
        url=pdf_url,
    )

    # Reconstruct full title if detail table stored only short subject
    if reg_num and "nomor" not in title.lower() and doc_type.lower() not in title.lower():
        full_title = f"{doc_type} UGM {reg_num} tentang {title}"
    else:
        full_title = title

    return {
        "title": clean_whitespace(full_title),
        "doc_type": doc_type,
        "regulation_number": clean_whitespace(reg_num),
        "year": year,
        "valid_time": eff_date,
        "effective_date": eff_date,
        "status": normalize_status(meta.get("status", ""), title=full_title),
        "related_regulations": extract_related_regulations(full_title),
        "version": extract_version(full_title, pdf_url),
        "source_url": page_url,
        "pdf_url": pdf_url,
        "wrapper_url": "",
        "source_name": source_name,
    }


def parse_html_pdf_links(html: str, page_url: str, source_name: str, default_doc_type: str) -> list[dict[str, Any]]:
    """Extract PDF regulation links from general WordPress content pages."""
    soup = BeautifulSoup(html, "lxml")
    container = soup.find("main") or soup.find("article") or soup
    items: list[dict[str, Any]] = []
    seen_pdfs: set[str] = set()

    for a in container.find_all("a", href=True):
        href = urljoin(page_url, a["href"].strip())
        if not urlparse(href).path.lower().endswith(".pdf"):
            continue
        if href in seen_pdfs:
            continue
        seen_pdfs.add(href)

        anchor_txt = clean_whitespace(a.get_text(" ", strip=True))
        if not anchor_txt or anchor_txt.lower() in ("klik di sini", "unduh", "download", "pdf", "lihat"):
            parent_tr = a.find_parent("tr")
            if parent_tr:
                anchor_txt = clean_whitespace(parent_tr.get_text(" ", strip=True))
            elif a.parent:
                anchor_txt = clean_whitespace(a.parent.get_text(" ", strip=True))
        if not anchor_txt or len(anchor_txt) < 5:
            anchor_txt = urlparse(href).path.rsplit("/", 1)[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")

        reg_num = extract_regulation_number(f"{anchor_txt} {href}")
        doc_type = normalize_doc_type(anchor_txt, url=href, default=default_doc_type)
        year = extract_year(title=anchor_txt, reg_num=reg_num, url=href)

        items.append(
            {
                "title": anchor_txt,
                "doc_type": doc_type,
                "regulation_number": reg_num,
                "year": year,
                "valid_time": "",
                "effective_date": "",
                "status": "Unknown",
                "related_regulations": extract_related_regulations(anchor_txt),
                "version": extract_version(anchor_txt, href),
                "source_url": page_url,
                "pdf_url": href,
                "wrapper_url": "",
                "source_name": source_name,
            }
        )
    return items


def parse_wp_media_items(media_json: list[dict[str, Any]], source_name: str, default_doc_type: str) -> list[dict[str, Any]]:
    """Convert WordPress /wp-json/wp/v2/media PDF items into standardized catalog entries."""
    items: list[dict[str, Any]] = []
    for entry in media_json:
        pdf_url = (entry.get("source_url") or "").strip()
        if not pdf_url or not urlparse(pdf_url).path.lower().endswith(".pdf"):
            continue
        raw_title = (entry.get("title") or {}).get("rendered") or ""
        title = clean_whitespace(raw_title)
        if not title:
            title = urlparse(pdf_url).path.rsplit("/", 1)[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")
        # Replace pure slug hyphens if title is a filename slug
        if "-" in title and " " not in title:
            title = title.replace("-", " ").replace("_", " ")

        wp_date = (entry.get("date") or "")[:10]
        reg_num = extract_regulation_number(f"{title} {pdf_url}")
        doc_type = normalize_doc_type(title, url=pdf_url, default=default_doc_type)
        year = extract_year(title=title, reg_num=reg_num, url=pdf_url, wp_date=wp_date)
        source_url = entry.get("link") or pdf_url

        items.append(
            {
                "title": title,
                "doc_type": doc_type,
                "regulation_number": reg_num,
                "year": year,
                "valid_time": wp_date if re.match(r"^\d{4}-\d{2}-\d{2}$", wp_date) else "",
                "effective_date": "",
                "status": "Unknown",
                "related_regulations": extract_related_regulations(title),
                "version": extract_version(title, pdf_url),
                "source_url": source_url,
                "pdf_url": pdf_url,
                "wrapper_url": "",
                "source_name": source_name,
            }
        )
    return items
