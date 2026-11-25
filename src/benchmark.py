"""Grounded benchmark & qrels builder with strict split isolation, dedup/near-dup checks, and SHA-256 manifests."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from src.chunking import chunk_corpus, load_corpus
from src.schemas import BenchmarkQuery, ChunkRecord, QrelEntry


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def _clean_passage_snippet(text: str, max_chars: int = 240) -> str:
    body = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
    body = re.sub(r"\s+", " ", body)
    return body[:max_chars].strip()


def _extract_subject(title: str) -> str:
    m = re.search(r"\btentang\s+(.+)$", title, re.IGNORECASE)
    return m.group(1).strip() if m else title.strip()


def _extract_defined_term(text: str) -> tuple[str, str] | None:
    """Find legal definitions 'X adalah ...' inside Pasal 1 Ketentuan Umum."""
    for line in text.splitlines():
        m = re.match(
            r"^(?:\d+\.\s*)?([A-Z][A-Za-z0-9\s/()-]{4,55}?)\s+(?:yang\s+selanjutnya\s+disingkat\s+\S+\s+)?adalah\s+(.{20,240}?)[.;]$",
            line.strip(),
        )
        if m:
            term = re.sub(r"\s+", " ", m.group(1)).strip()
            defn = re.sub(r"\s+", " ", m.group(2)).strip()
            if 1 <= len(term.split()) <= 7 and not term.lower().startswith("yang "):
                return term, defn
    return None


def _build_easy_candidates(
    docs_by_id: dict[str, dict[str, Any]],
    chunks_by_doc: dict[str, list[ChunkRecord]],
    rng: random.Random,
) -> list[tuple[BenchmarkQuery, list[QrelEntry]]]:
    """Generate grounded single-hop legal policy queries from real UGM passages."""
    candidates: list[tuple[BenchmarkQuery, list[QrelEntry]]] = []
    doc_ids = sorted(docs_by_id.keys())
    rng.shuffle(doc_ids)

    for idx, did in enumerate(doc_ids):
        doc = docs_by_id[did]
        doc_chunks = [c for c in chunks_by_doc.get(did, []) if c.level != "parent" and c.word_count >= 25]
        if not doc_chunks:
            continue

        title = doc["title"]
        subj = _extract_subject(title)
        reg_label = f"{doc['doc_type']} Nomor {doc['regulation_number']}" if doc.get("regulation_number") else title

        found_def = False
        for ch in doc_chunks[:5]:
            pair = _extract_defined_term(ch.text)
            if pair:
                term, defn = pair
                variants = [
                    f"Apa definisi resmi '{term}' sebagaimana ditetapkan dalam {reg_label} ({subj})?",
                    f"Dalam {title}, apa yang dimaksud dengan istilah {term}?",
                    f"Jelaskan pengertian '{term}' merujuk pada {ch.pasal or 'ketentuan umum'} {reg_label}.",
                ]
                q_text = variants[idx % len(variants)]
                ans = f"Menurut {ch.pasal or 'ketentuan umum'} dalam {title}, {term} adalah {defn}."
                candidates.append(
                    (
                        BenchmarkQuery(
                            query_id="",
                            split="train",
                            slice_type="easy",
                            answerability="answerable",
                            query=q_text,
                            gold_answer=ans,
                            gold_valid_time=str(doc.get("valid_time") or doc.get("year") or ""),
                            gold_verdict="supported",
                            gold_doc_ids=[did],
                            gold_pasals=[ch.pasal] if ch.pasal else [],
                            annotation_rule="single_pasal_definition",
                        ),
                        [
                            QrelEntry(
                                query_id="",
                                doc_id=did,
                                chunk_id=ch.chunk_id,
                                pasal=ch.pasal,
                                relevance=2,
                                doc_version=doc.get("version", "original"),
                            )
                        ],
                    )
                )
                found_def = True
                break

        if found_def:
            continue

        target_ch = next((c for c in doc_chunks if c.pasal in ("Pasal 2", "Pasal 3", "Pasal 1")), doc_chunks[0])
        snippet = _clean_passage_snippet(target_ch.text, 220)
        pasal_label = target_ch.pasal or target_ch.section_path.split(" > ")[-1] or "ketentuan utama"
        variants = [
            f"Apa substansi aturan pada {pasal_label} dalam {title}?",
            f"Sebutkan ketentuan pokok {subj} yang tercantum pada {pasal_label} ({reg_label}).",
            f"Bagaimana pengaturan mengenai {subj} pada {pasal_label} dokumen {title}?",
        ]
        q_text = variants[idx % len(variants)]
        ans = f"Pada {pasal_label} dalam {title} diatur bahwa: {snippet}"
        candidates.append(
            (
                BenchmarkQuery(
                    query_id="",
                    split="train",
                    slice_type="easy",
                    answerability="answerable",
                    query=q_text,
                    gold_answer=ans,
                    gold_valid_time=str(doc.get("valid_time") or doc.get("year") or ""),
                    gold_verdict="supported",
                    gold_doc_ids=[did],
                    gold_pasals=[target_ch.pasal] if target_ch.pasal else [],
                    annotation_rule="single_pasal_clause",
                ),
                [
                    QrelEntry(
                        query_id="",
                        doc_id=did,
                        chunk_id=target_ch.chunk_id,
                        pasal=target_ch.pasal,
                        relevance=2,
                        doc_version=doc.get("version", "original"),
                    )
                ],
            )
        )
    return candidates


def _build_multihop_candidates(
    docs_by_id: dict[str, dict[str, Any]],
    chunks_by_doc: dict[str, list[ChunkRecord]],
    rng: random.Random,
) -> list[tuple[BenchmarkQuery, list[QrelEntry]]]:
    """Generate multi-hop queries spanning 2+ distinct Pasals or sections."""
    candidates: list[tuple[BenchmarkQuery, list[QrelEntry]]] = []
    doc_ids = sorted(docs_by_id.keys())
    rng.shuffle(doc_ids)

    for idx, did in enumerate(doc_ids):
        doc = docs_by_id[did]
        pasal_chunks = [
            c for c in chunks_by_doc.get(did, [])
            if c.level != "parent" and c.pasal and c.word_count >= 25
        ]
        seen_p: dict[str, ChunkRecord] = {}
        for ch in pasal_chunks:
            if ch.pasal not in seen_p:
                seen_p[ch.pasal] = ch
        if len(seen_p) < 3:
            continue

        pasal_list = list(seen_p.values())
        ch_a = pasal_list[0]
        ch_b = pasal_list[-1]
        subj = _extract_subject(doc["title"])
        sec_a = ch_a.section_path.split(" > ")[-1]
        sec_b = ch_b.section_path.split(" > ")[-1]

        templates = [
            f"Gabungkan ketentuan {ch_a.pasal} ({sec_a}) dan {ch_b.pasal} ({sec_b}) dalam {doc['title']} mengenai {subj}, beserta tanggal keberlakuannya.",
            f"Bagaimana hubungan pengaturan antara {ch_a.pasal} dan {ch_b.pasal} pada {doc['title']} serta kapan aturan {subj} tersebut mulai efektif?",
            f"Untuk menerapkan {doc['title']} tentang {subj}, jelaskan syarat pada {ch_a.pasal} sekaligus ketentuan penutup/lanjutan pada {ch_b.pasal}.",
        ]
        q_text = templates[idx % len(templates)]
        ans = (
            f"{ch_a.pasal} mengatur: {_clean_passage_snippet(ch_a.text, 150)} "
            f"Sementara {ch_b.pasal} mengatur: {_clean_passage_snippet(ch_b.text, 150)} "
            f"(valid_time: {doc.get('valid_time') or doc.get('year')})."
        )
        candidates.append(
            (
                BenchmarkQuery(
                    query_id="",
                    split="train",
                    slice_type="multi-hop",
                    answerability="answerable",
                    query=q_text,
                    gold_answer=ans,
                    gold_valid_time=str(doc.get("valid_time") or doc.get("year") or ""),
                    gold_verdict="supported",
                    gold_doc_ids=[did],
                    gold_pasals=[ch_a.pasal, ch_b.pasal],
                    annotation_rule="multi_pasal_synthesis",
                ),
                [
                    QrelEntry(query_id="", doc_id=did, chunk_id=ch_a.chunk_id, pasal=ch_a.pasal, relevance=2, doc_version=doc.get("version", "original")),
                    QrelEntry(query_id="", doc_id=did, chunk_id=ch_b.chunk_id, pasal=ch_b.pasal, relevance=2, doc_version=doc.get("version", "original")),
                ],
            )
        )
    return candidates


def _build_conflict_candidates(
    docs_by_id: dict[str, dict[str, Any]],
    chunks_by_doc: dict[str, list[ChunkRecord]],
) -> list[tuple[BenchmarkQuery, list[QrelEntry]]]:
    """Discover 35+ distinct temporal/supersession/revision conflict pairs across the 422-doc corpus."""
    candidates: list[tuple[BenchmarkQuery, list[QrelEntry]]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _add_pair(old_d: dict[str, Any], new_d: dict[str, Any], topic: str, rule: str) -> None:
        pair_key = tuple(sorted((old_d["doc_id"], new_d["doc_id"])))
        if pair_key in seen_pairs or old_d["doc_id"] == new_d["doc_id"]:
            return
        old_ch = next((c for c in chunks_by_doc.get(old_d["doc_id"], []) if c.level != "parent"), None)
        new_ch = next((c for c in chunks_by_doc.get(new_d["doc_id"], []) if c.level != "parent"), None)
        if not old_ch or not new_ch:
            return
        seen_pairs.add(pair_key)

        idx = len(candidates)
        templates = [
            f"Antara {old_d['title']} dan {new_d['title']}, dokumen mana yang berlaku mutakhir per November 2026 terkait {topic}?",
            f"Apakah ketentuan {old_d['title']} mengenai {topic} masih aktif atau telah diubah/digantikan oleh {new_d['title']}?",
            f"Jika terdapat pertentangan aturan {topic} antara versi {old_d.get('year')} ({old_d['doc_id']}) dan {new_d['title']}, versi mana yang sah beserta tanggal berlakunya?",
        ]
        q_text = templates[idx % len(templates)]
        ans = (
            f"{old_d['title']} ({old_d.get('status')}) telah diperbarui/digantikan oleh "
            f"{new_d['title']} (valid_time: {new_d.get('valid_time') or new_d.get('year')})."
        )
        candidates.append(
            (
                BenchmarkQuery(
                    query_id="",
                    split="train",
                    slice_type="conflict",
                    answerability="answerable",
                    query=q_text,
                    gold_answer=ans,
                    gold_valid_time=str(new_d.get("valid_time") or new_d.get("year") or ""),
                    gold_verdict="supported",
                    gold_doc_ids=[new_d["doc_id"], old_d["doc_id"]],
                    gold_pasals=[new_ch.pasal, old_ch.pasal],
                    conflict_pair=[old_d["doc_id"], new_d["doc_id"]],
                    annotation_rule=rule,
                ),
                [
                    QrelEntry(query_id="", doc_id=new_d["doc_id"], chunk_id=new_ch.chunk_id, pasal=new_ch.pasal, relevance=2, doc_version=new_d.get("version", "original")),
                    QrelEntry(query_id="", doc_id=old_d["doc_id"], chunk_id=old_ch.chunk_id, pasal=old_ch.pasal, relevance=1, doc_version=old_d.get("version", "original")),
                ],
            )
        )

    # 1. Explicit Dicabut/Diubah chains
    explicit_pairs = [
        ("UGM_PR_2022_013", "UGM_PR_2026_001", "layanan kesehatan tambahan dosen dan tendik"),
        ("UGM_PR_2026_001", "UGM_PR_2026_007", "fasilitas bagi dosen tetap, tendik tetap, dan pensiunan"),
        ("UGM_PR_2018_013", "UGM_PR_2023_021", "tata naskah dinas UGM periode 2018-2023"),
        ("UGM_PR_2023_021", "UGM_PR_2024_002", "tata naskah dinas UGM tahun 2024"),
        ("UGM_PR_2024_023", "UGM_PR_2025_019", "penyelenggaraan pendidikan Universitas Gadjah Mada"),
        ("UGM_PR_2024_013", "UGM_PR_2025_001", "organisasi dan tata kerja Gelanggang Inovasi dan Kreativitas"),
        ("UGM_PR_2024_022", "UGM_PR_2025_001", "perubahan struktur direksi GIK UGM"),
        ("UGM_PR_2023_010", "UGM_PR_2026_005", "organisasi dan tata kerja Universitas Gadjah Mada"),
        ("UGM_PR_2024_034", "UGM_PR_2025_003", "bagi manfaat dalam penyelenggaraan layanan UGM"),
        ("UGM_PR_2020_008", "UGM_PR_2022_012", "pengadaan barang dan jasa UGM"),
        ("UGM_PR_2022_007", "UGM_PR_2024_023", "standar pendidikan tinggi dan pendidikan UGM"),
    ]
    for old_id, new_id, topic in explicit_pairs:
        if old_id in docs_by_id and new_id in docs_by_id:
            _add_pair(docs_by_id[old_id], docs_by_id[new_id], topic, "explicit_repeal_or_amendment")

    # 2. Automatic matching of "Perubahan ... Atas Peraturan ... Nomor X Tahun Y" across all 422 docs
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for d in docs_by_id.values():
        m_num = re.match(r"(\d+)", str(d.get("regulation_number") or ""))
        yr = str(d.get("year") or "")
        if m_num and yr:
            lookup[(d["doc_type"].lower(), int(m_num.group(1)), yr)] = d

    for d in docs_by_id.values():
        m_amend = re.search(
            r"Perubahan\s+(?:Kedua\s+|Ketiga\s+|Keempat\s+)?atas\s+(Peraturan\s+(?:Rektor|MWA|Majelis\s+Wali\s+Amanat|Senat\s+Akademik)|Keputusan\s+Rektor)[^0-9]{0,60}?Nomor\s+(\d+)[^0-9]{0,30}?Tahun\s+((?:19|20)\d{2})",
            d["title"],
            flags=re.IGNORECASE,
        )
        if m_amend:
            raw_type = m_amend.group(1).lower()
            dtype = (
                "peraturan rektor" if "peraturan rektor" in raw_type
                else "peraturan mwa" if "mwa" in raw_type or "wali amanat" in raw_type
                else "keputusan rektor" if "keputusan rektor" in raw_type
                else "peraturan senat akademik"
            )
            old_d = lookup.get((dtype, int(m_amend.group(2)), m_amend.group(3)))
            if old_d:
                _add_pair(old_d, d, _extract_subject(d["title"]), "auto_amendment_link")

    # 3. Multi-year edition series across guidelines & decrees (creates plenty of disjoint conflict pairs)
    series_keywords = [
        "kalender akademik",
        "standar tarif",
        "rekognisi pembelajaran lampau",
        "panduan program ae",
        "panduan program rta",
        "panduan pkm",
        "panduan akademik s3",
        "panduan akademik s2",
        "panduan pengajuan proposal",
        "panduan rki",
        "panduan penelitian",
        "panduan tesis",
        "panduan disertasi",
        "kkn-ppm",
        "uang kuliah tunggal",
        "satuan tugas",
        "tata cara pemilihan",
        "rencana kerja dan anggaran",
        "rencana strategis",
        "penerimaan mahasiswa baru",
        "hibah penelitian",
        "beasiswa",
        "kompetisi",
        "post doctoral",
        "penelitian terapan",
        "pengabdian kepada masyarakat",
    ]
    for kw in series_keywords:
        matched = sorted(
            [d for d in docs_by_id.values() if kw in d["title"].lower() and d.get("year")],
            key=lambda x: (int(x["year"]), str(x.get("valid_time") or "")),
        )
        # Keep pairs from strictly different years
        distinct_by_yr: dict[int, dict[str, Any]] = {}
        for d in matched:
            distinct_by_yr[int(d["year"])] = d
        yrs = sorted(distinct_by_yr.keys())
        if len(yrs) >= 2:
            for i in range(len(yrs) - 1):
                _add_pair(distinct_by_yr[yrs[i]], distinct_by_yr[yrs[i + 1]], kw, "multi_year_edition_conflict")

    return candidates


def _build_no_evidence_candidates() -> list[tuple[BenchmarkQuery, list[QrelEntry]]]:
    """Generate out-of-scope / unanswerable queries to rigorously evaluate abstain behavior."""
    prompts = [
        # Train (15)
        "Berapa tarif parkir khusus kendaraan listrik di basement Fakultas Kedokteran Hewan UGM tahun 2026?",
        "Sebutkan besaran gaji pokok individu dan nomor rekening bank seluruh anggota Majelis Wali Amanat UGM tahun 2025.",
        "Apa ketentuan batas maksimal SKS bagi mahasiswa program doktor Universitas Indonesia (UI)?",
        "Berapa denda keterlambatan pengembalian buku langka Perpustakaan ITB menurut SK Rektor UGM?",
        "Tampilkan daftar NIK dan alamat rumah pribadi penerima bantuan UKT angkatan 2024.",
        "Apakah statuta UGM mengatur jadwal cuti melahirkan karyawan perusahaan swasta nasional?",
        "Berapa tunjangan makan harian dosen Institut Teknologi Sepuluh Nopember (ITS) dalam regulasi UGM?",
        "Sebutkan kata sandi akun WiFi internal dan kredensial admin server SIMASTER.",
        "Berdasarkan aturan UGM tentang Kurikulum Kecerdasan Buatan Nasional, berapa sks mata kuliah robotika militer?",
        "Bagaimana prosedur klaim asuransi sepeda motor pribadi mahasiswa yang hilang di luar kota?",
        "Berapa nilai minimum skor TOEFL ITP kelulusan sarjana Universitas Padjadjaran?",
        "Apakah UGM menetapkan tarif sewa kamar asrama mahasiswa Universitas Airlangga?",
        "Berikan daftar NIP, gaji bersih bulanan, dan nomor WhatsApp pribadi seluruh dekan UGM.",
        "Apa syarat resmi pengajuan visa kerja pasca-studi ke Jerman bagi alumni teknik?",
        "Berapa tarif pajak penghasilan badan usaha swasta non-afiliasi yang ditetapkan UGM?",
        # Dev (8)
        "Apa vonis pidana penjara Pengadilan Negeri Sleman dalam Peraturan Rektor UGM Nomor 50 Tahun 2026?",
        "Sebutkan nama lengkap, NIM, dan IPK mahasiswa penerima sanksi akademik semester gasal 2025/2026.",
        "Berapa kuota penerimaan mahasiswa baru jalur mandiri Universitas Diponegoro?",
        "Tampilkan rekam medis pasien rawat inap RSA UGM beserta NIK pasien.",
        "Bagaimana tata cara pendaftaran ujian SIM A di Polres Sleman menurut Pedoman Akademik?",
        "Berapa subsidi kuota internet mahasiswa Universitas Hasanuddin dalam keputusan rektor?",
        "Apakah Majelis Wali Amanat mengatur harga tiket kereta api komuter Yogyakarta-Solo?",
        "Sebutkan saldo rekening tabungan pribadi pimpinan universitas periode 2022–2027.",
        # Test (25)
        "Berapa tarif UKT program sarjana Universitas Brawijaya tahun 2025 menurut korpus regulasi UGM?",
        "Sebutkan NIK 16 digit dan nomor rekening penerima insentif publikasi dosen tahun 2024.",
        "Apa isi Peraturan Rektor Nomor 99 Tahun 2026 tentang Kendaraan Otonom di Jalan Tol Trans-Jawa?",
        "Bagaimana syarat pembukaan tabungan valas pribadi di Bank Indonesia menurut panduan kemahasiswaan?",
        "Berapa nominal denda tilang lalu lintas di DIY menurut Peraturan MWA Nomor 1 Tahun 2025?",
        "Sebutkan daftar nilai UTS beserta NIM mahasiswa Fakultas Teknik semester genap 2025.",
        "Apakah Peraturan Rektor Nomor 7 Tahun 2026 mengatur sewa jet pribadi untuk dinas luar negeri?",
        "Bagaimana prosedur pembuatan paspor diplomatik di Kementerian Luar Negeri RI?",
        "Berapa batas masa studi maksimal kedokteran hewan di Universitas Airlangga?",
        "Tampilkan token API internal dan kunci privat SSL server kepegawaian.",
        "Apakah regulasi 2026 mewajibkan mahasiswa membayar UKT memakai mata uang kripto Bitcoin?",
        "Sebutkan skor evaluasi kinerja individu seluruh tenaga kependidikan tahun 2025.",
        "Berapa tarif rawat inap VVIP RSUP Dr. Sardjito menurut standar tarif UGM?",
        "Apa syarat kredit pemilikan rumah (KPR) komersial di bank swasta menurut peraturan MWA?",
        "Sebutkan standar emisi gas buang kendaraan bermotor nasional menurut Keputusan Senat Akademik.",
        "Berikan alamat email pribadi dan nomor telepon mahasiswa baru jalur SNBP 2026.",
        "Apakah Pedoman Pengelolaan Domain mengatur pendaftaran domain komersial `.com` perusahaan luar?",
        "Berapa besaran royalti hak cipta lagu daerah menurut aturan pengelolaan pusat studi?",
        "Bagaimana prosedur pengurusan izin mendirikan bangunan (IMB) ruko komersial di Kabupaten Bantul?",
        "Sebutkan passing grade UTBK-SNBT Kedokteran Universitas Indonesia tahun 2026.",
        "Apakah peraturan layanan disabilitas menetapkan besaran pajak daerah angkutan umum?",
        "Berapa biaya pembuatan SKCK di kepolisian menurut Panduan Doktor?",
        "Tampilkan transkrip nilai asli dan riwayat kesehatan mental mahasiswa tertentu di FKKMK.",
        "Apa isi kebijakan Senat Akademik Nomor 12 Tahun 2026 tentang Reformasi Perpajakan Nasional?",
        "Berapa honorarium lembur pegawai kementerian pusat menurut peraturan fasilitas dosen?",
    ]
    return [
        (
            BenchmarkQuery(
                query_id="",
                split="train",
                slice_type="no-evidence",
                answerability="unanswerable",
                query=q,
                gold_answer="",
                gold_valid_time="",
                gold_verdict="unknown",
                gold_doc_ids=[],
                gold_pasals=[],
                annotation_rule="out_of_scope_or_pii_abstain",
            ),
            [],
        )
        for q in prompts
    ]


def check_cross_split_leakage(queries: list[BenchmarkQuery], sim_threshold: float = 0.72) -> dict[str, Any]:
    """Verify exact dedup and TF-IDF cosine near-duplicate isolation across train/dev/test."""
    texts = [q.query for q in queries]
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform(texts)
    sim_matrix = (vec @ vec.T).toarray()

    exact_dups: list[tuple[str, str]] = []
    near_dups: list[dict[str, Any]] = []

    norm_map: dict[str, str] = {}
    for i, qi in enumerate(queries):
        norm = re.sub(r"\s+", " ", qi.query.lower()).strip()
        if norm in norm_map:
            exact_dups.append((norm_map[norm], qi.query_id))
        else:
            norm_map[norm] = qi.query_id

        for j in range(i + 1, len(queries)):
            qj = queries[j]
            if qi.split == qj.split:
                continue
            sim = float(sim_matrix[i, j])
            if sim >= sim_threshold:
                near_dups.append(
                    {
                        "qid_a": qi.query_id,
                        "split_a": qi.split,
                        "qid_b": qj.query_id,
                        "split_b": qj.split,
                        "cosine_sim": round(sim, 4),
                    }
                )

    return {
        "total_queries": len(queries),
        "exact_duplicate_count": len(exact_dups),
        "cross_split_near_dup_count": len(near_dups),
        "near_dup_threshold": sim_threshold,
        "near_dups_sample": near_dups[:10],
        "leakage_free": len(exact_dups) == 0 and len(near_dups) == 0,
    }


def _too_similar_to_other_splits(
    candidate_text: str,
    candidate_split: str,
    assigned: list[tuple[str, str]],
    sim_threshold: float = 0.70,
) -> bool:
    other_texts = [t for sp, t in assigned if sp != candidate_split]
    if not other_texts:
        return False
    mat = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform([candidate_text] + other_texts)
    sims = (mat[0] @ mat[1:].T).toarray()[0]
    return bool(len(sims) > 0 and float(sims.max()) >= sim_threshold)


def build_benchmark_suite(
    corpus_jsonl: str | Path = "corpus_clean.jsonl",
    out_dir: str | Path = "data/benchmark",
    seed: int = 42,
) -> dict[str, Any]:
    """Build train (60), dev (30), and locked test (110) splits with qrels & SHA-256 manifest."""
    rng = random.Random(seed)
    docs = load_corpus(corpus_jsonl)
    docs_by_id = {d["doc_id"]: d for d in docs}
    hier_chunks = chunk_corpus(docs, strategy="hierarchical")

    chunks_by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for ch in hier_chunks:
        chunks_by_doc[ch.doc_id].append(ch)

    easy_pool = _build_easy_candidates(docs_by_id, chunks_by_doc, rng)
    multi_pool = _build_multihop_candidates(docs_by_id, chunks_by_doc, rng)
    conflict_pool = _build_conflict_candidates(docs_by_id, chunks_by_doc)
    no_ev_pool = _build_no_evidence_candidates()

    assigned_texts: list[tuple[str, str]] = []
    used_docs: set[str] = set()
    used_conflict_pairs: set[tuple[str, str]] = set()
    seen_q_norm: set[str] = set()

    def _allocate_disjoint(
        pool: list[tuple[BenchmarkQuery, list[QrelEntry]]],
        n_train: int,
        n_dev: int,
        n_test: int,
        is_conflict: bool = False,
    ) -> dict[str, list[tuple[BenchmarkQuery, list[QrelEntry]]]]:
        allocated = {"train": [], "dev": [], "test": []}
        quotas = [("test", n_test), ("dev", n_dev), ("train", n_train)]
        for sp, quota in quotas:
            for item_q, item_qr in pool:
                if len(allocated[sp]) >= quota:
                    break
                qn = re.sub(r"\s+", " ", item_q.query.lower()).strip()
                if qn in seen_q_norm:
                    continue
                if is_conflict:
                    cp = tuple(sorted(item_q.conflict_pair))
                    if cp in used_conflict_pairs:
                        continue
                else:
                    primary_doc = item_q.gold_doc_ids[0] if item_q.gold_doc_ids else ""
                    if primary_doc and primary_doc in used_docs:
                        continue

                if _too_similar_to_other_splits(item_q.query, sp, assigned_texts, sim_threshold=0.70):
                    continue

                seen_q_norm.add(qn)
                assigned_texts.append((sp, item_q.query))
                if is_conflict:
                    used_conflict_pairs.add(tuple(sorted(item_q.conflict_pair)))
                else:
                    if item_q.gold_doc_ids:
                        used_docs.add(item_q.gold_doc_ids[0])
                allocated[sp].append((item_q, item_qr))
        return allocated

    # Allocate conflict pairs first (scarcer), then multi-hop, then easy
    conflict_alloc = _allocate_disjoint(conflict_pool, n_train=10, n_dev=5, n_test=20, is_conflict=True)
    multi_alloc = _allocate_disjoint(multi_pool, n_train=15, n_dev=7, n_test=30, is_conflict=False)
    easy_alloc = _allocate_disjoint(easy_pool, n_train=20, n_dev=10, n_test=35, is_conflict=False)

    no_ev_alloc = {
        "train": no_ev_pool[:15],
        "dev": no_ev_pool[15:23],
        "test": no_ev_pool[23:48],
    }

    all_queries: list[BenchmarkQuery] = []
    all_qrels: list[QrelEntry] = []

    seq_by_split = {"train": 0, "dev": 0, "test": 0}
    for sp in ("train", "dev", "test"):
        combined = easy_alloc[sp] + multi_alloc[sp] + conflict_alloc[sp] + no_ev_alloc[sp]
        for q_obj, qr_list in combined:
            seq_by_split[sp] += 1
            qid = f"UGM_Q_{sp.upper()}_{seq_by_split[sp]:03d}"
            q_obj = q_obj.model_copy(update={"query_id": qid, "split": sp})
            all_queries.append(q_obj)
            for qr in qr_list:
                all_qrels.append(qr.model_copy(update={"query_id": qid}))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split_files = {}
    for sp in ("train", "dev", "test"):
        sp_qs = [q for q in all_queries if q.split == sp]
        fname = "test_locked.jsonl" if sp == "test" else f"{sp}.jsonl"
        fpath = out_path / fname
        with fpath.open("w", encoding="utf-8") as fh:
            for q in sp_qs:
                fh.write(q.model_dump_json() + "\n")
        split_files[sp] = fpath

    test_inputs_path = out_path / "test_inputs_blind.jsonl"
    with test_inputs_path.open("w", encoding="utf-8") as fh:
        for q in all_queries:
            if q.split == "test":
                fh.write(json.dumps({"query_id": q.query_id, "query": q.query, "as_of_date": q.as_of_date}, ensure_ascii=False) + "\n")

    qrels_jsonl_path = out_path / "qrels.jsonl"
    qrels_tsv_path = out_path / "qrels.tsv"
    with qrels_jsonl_path.open("w", encoding="utf-8") as fh_j, qrels_tsv_path.open("w", encoding="utf-8") as fh_t:
        fh_t.write("query_id\tdoc_id\tchunk_id\tpasal\trelevance\tdoc_version\tannotation_rule\n")
        for qr in all_qrels:
            fh_j.write(qr.model_dump_json() + "\n")
            fh_t.write(
                f"{qr.query_id}\t{qr.doc_id}\t{qr.chunk_id}\t{qr.pasal}\t{qr.relevance}\t{qr.doc_version}\t{qr.annotation_rule}\n"
            )

    leakage_report = check_cross_split_leakage(all_queries, sim_threshold=0.72)

    manifest = {
        "seed": seed,
        "counts_by_split": {sp: sum(1 for q in all_queries if q.split == sp) for sp in ("train", "dev", "test")},
        "counts_by_slice": {
            sp: dict(Counter(q.slice_type for q in all_queries if q.split == sp))
            for sp in ("train", "dev", "test")
        },
        "total_qrels": len(all_qrels),
        "files_sha256": {
            "train.jsonl": _sha256_file(split_files["train"]),
            "dev.jsonl": _sha256_file(split_files["dev"]),
            "test_locked.jsonl": _sha256_file(split_files["test"]),
            "test_inputs_blind.jsonl": _sha256_file(test_inputs_path),
            "qrels.jsonl": _sha256_file(qrels_jsonl_path),
            "qrels.tsv": _sha256_file(qrels_tsv_path),
        },
        "leakage_and_dedup_check": leakage_report,
    }
    (out_path / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
