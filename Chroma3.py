import sys
from datetime import datetime
from pathlib import Path
from pypdf import PdfReader

# ------------------------------------------------------------------
# KONFIGURASI (ubah sesuai kebutuhan)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FOLDER_PDF = BASE_DIR / "dokumen"
FOLDER_DB = BASE_DIR / "chroma_db"
FOLDER_HASIL = BASE_DIR / "hasil"
FILE_HASIL = FOLDER_HASIL / "Lab3_test.txt"

MODEL_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# Dua koleksi yang dibandingkan: (nama koleksi, ukuran chunk, overlap)
KONFIGURASI_KOLEKSI = [
    ("pedoman_500", 500, 75),
    ("pedoman_1000", 1000, 150),
]
NILAI_K = (1, 3, 5)

UJI = [
    ("Apa yang menjadi salah satu persoalan dalam kehidupan demokrasi akibat aktivitas pembelaan politik tanpa batas?", "buzzer politik"),

    ("Apa yang seharusnya menjadi dasar masyarakat dalam memberikan suara saat pemilu?", "pertimbangan yang matang"),

    ("Apa yang berhak dilakukan masyarakat ketika pemerintah melakukan kesalahan?", "mengkritik"),

    ("Apa yang sering diutamakan sebagian pemilih daripada logika?", "fanatisme"),

    ("Apa yang harus dipelajari masyarakat sebelum menentukan pilihan dalam pemilu?", "rekam jejak calon pemimpin"),

    ("Apa yang menjadi tanggung jawab moral dari kebebasan memilih dalam demokrasi?", "mempertimbangkan pilihan secara rasional"),

    ("Apa yang seharusnya menjadi salah satu alat untuk memperbaiki pemerintahan?", "kritik"),

    ("Apa yang berhak diketahui publik jika seseorang mendapatkan keuntungan finansial dari aktivitas buzzer?", "transparansi"),

    ("Apa yang dibutuhkan masyarakat ketika mengeluhkan keadaan ekonomi kepada pemerintah?", "solusi"),

    ("Apa yang seharusnya menjadi prioritas utama dalam setiap keputusan seorang presiden?", "kepentingan rakyat"),

    ("Apa yang harus dilakukan terhadap kebijakan pemerintah jika tidak memberikan hasil yang diharapkan?", "dievaluasi"),

    ("Apa yang harus dimiliki seorang presiden untuk menghadapi kesalahan dalam kepemimpinannya?", "keberanian untuk bertanggung jawab"),

    ("Apa yang dapat memengaruhi harga kebutuhan pokok, pendidikan, kesehatan, dan lapangan pekerjaan?", "Kebijakan yang dibuat seorang presiden"),

    ("Apa yang dibutuhkan seorang presiden selain kemampuan menyampaikan pidato?", "berpikir jernih"),

    ("Apa yang dibutuhkan Indonesia dari seorang pemimpin untuk membangun masa depan bangsa?", "hasil nyata"),
]


# ------------------------------------------------------------------
# FUNGSI DASAR
# ------------------------------------------------------------------
def daftar_pdf():
    if not FOLDER_PDF.exists():
        FOLDER_PDF.mkdir(parents=True)
        sys.exit(f"Folder '{FOLDER_PDF}' baru dibuat. Isi dengan 2-3 PDF, lalu jalankan lagi.")
    pdfs = sorted(FOLDER_PDF.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Tidak ada file .pdf di '{FOLDER_PDF}'. Tambahkan PDF dulu.")
    return pdfs


def baca_pdf(path):
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def potong(teks, ukuran=800, tumpang=100):
    """Pemotongan ukuran tetap dengan overlap."""
    if tumpang >= ukuran:
        raise ValueError("tumpang harus lebih kecil dari ukuran")
    hasil, mulai = [], 0
    while mulai < len(teks):
        hasil.append(teks[mulai:mulai + ukuran])
        if mulai + ukuran >= len(teks):
            break
        mulai += ukuran - tumpang
    return [h for h in hasil if h.strip()]


def buat_embedding():
    from chromadb.utils import embedding_functions

    print(f"Memuat model embedding '{MODEL_EMBEDDING}' "
          "(pertama kali akan mengunduh ratusan MB)...")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_EMBEDDING)


def bangun_indeks(client, emb, nama, ukuran, tumpang, teks_pdf):
    """Buat koleksi baru (yang lama dihapus dulu). Teks PDF dibaca sekali saja
    dan dipakai ulang untuk semua koleksi."""
    try:
        client.delete_collection(nama)
    except Exception:
        pass
    koleksi = client.create_collection(
        name=nama, embedding_function=emb,
        metadata={"hnsw:space": "cosine"})  # jarak cosine

    ringkasan = []
    for pdf, teks in teks_pdf:
        chunks = potong(teks, ukuran, tumpang)
        if not chunks:
            ringkasan.append((pdf.name, 0))
            continue
        koleksi.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))],
        )
        ringkasan.append((pdf.name, len(chunks)))
    return koleksi, ringkasan


def evaluasi(koleksi, uji):
    """Satu query per pertanyaan (top-k maksimum), lalu hitung Hit@k dan MRR.
    Mengembalikan (rincian per pertanyaan, hit@k, mrr)."""
    k_maks = max(NILAI_K)
    semua_teks = " ".join(koleksi.get(include=["documents"])["documents"]).lower()

    rincian = []
    for pertanyaan, kunci in uji:
        hasil = koleksi.query(query_texts=[pertanyaan], n_results=k_maks)
        docs = hasil["documents"][0]
        metas = hasil["metadatas"][0]
        jarak = hasil["distances"][0]

        peringkat = None  # peringkat chunk benar pertama (1-based)
        for r, d in enumerate(docs, start=1):
            if kunci.lower() in d.lower():
                peringkat = r
                break

        rincian.append({
            "pertanyaan": pertanyaan,
            "kunci": kunci,
            "peringkat": peringkat,
            "kunci_ada_di_korpus": kunci.lower() in semua_teks,
            "top1_meta": metas[0] if metas else None,
            "top1_jarak": jarak[0] if jarak else None,
            "top1_doc": docs[0] if docs else "",
        })

    n = len(uji)
    hit = {k: sum(1 for r in rincian if r["peringkat"] and r["peringkat"] <= k) / n
           for k in NILAI_K}
    mrr = sum(1 / r["peringkat"] for r in rincian if r["peringkat"]) / n
    return rincian, hit, mrr


def uji_valid():
    valid = [(q, k) for q, k in UJI
             if not q.strip().endswith("...") and not k.strip().endswith("...")]
    if len(valid) < len(UJI):
        print(f"PERINGATAN: {len(UJI) - len(valid)} pasangan masih placeholder ('...') dan dilewati.")
    if not valid:
        sys.exit("Daftar UJI kosong. Isi pasangan (pertanyaan, kata kunci) di bagian atas file.")
    if len(valid) < 15:
        print(f"PERINGATAN: UJI baru berisi {len(valid)} pasangan (Lab meminta 15). "
              "Lengkapi agar hasilnya lebih bermakna.")
    return valid


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    import chromadb

    garis = "=" * 70
    uji = uji_valid()
    pdfs = daftar_pdf()

    print(f"Membaca {len(pdfs)} PDF...")
    teks_pdf = []
    for pdf in pdfs:
        teks = baca_pdf(pdf)
        if not teks.strip():
            print(f"PERINGATAN: {pdf.name} tidak berisi teks (PDF scan?), dilewati.")
            continue
        teks_pdf.append((pdf, teks))
    if not teks_pdf:
        sys.exit("Tidak ada PDF yang berisi teks.")

    emb = buat_embedding()
    client = chromadb.PersistentClient(path=str(FOLDER_DB))

    baris = [garis, "LAB 3 - UKUR HIT@K PADA DUA KOLEKSI", garis,
             f"Diperbarui : {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Model      : {MODEL_EMBEDDING}",
             f"Jumlah uji : {len(uji)} pertanyaan", ""]

    hasil_semua = []  # (nama, ukuran, tumpang, rincian, hit, mrr)
    for nama, ukuran, tumpang in KONFIGURASI_KOLEKSI:
        print("\n" + garis)
        print(f"KOLEKSI {nama} (chunk={ukuran}, overlap={tumpang})")
        print(garis)
        koleksi, ringkasan = bangun_indeks(client, emb, nama, ukuran, tumpang, teks_pdf)
        for nm, n in ringkasan:
            print(f"  {nm} -> {n} chunk")
        print(f"  Total: {koleksi.count()} chunk")

        rincian, hit, mrr = evaluasi(koleksi, uji)
        for k in NILAI_K:
            print(f"  Hit@{k} = {hit[k]:.2f}")
        print(f"  MRR   = {mrr:.2f}")
        hasil_semua.append((nama, ukuran, tumpang, ringkasan, koleksi.count(), rincian, hit, mrr))

    # --- Tabel perbandingan ---
    print("\n" + garis)
    print("PERBANDINGAN")
    print(garis)
    kepala = f"{'Koleksi':<16}{'Chunk':>7}" + "".join(f"{'Hit@' + str(k):>9}" for k in NILAI_K) + f"{'MRR':>8}"
    print(kepala)
    baris += ["PERBANDINGAN", kepala]
    for nama, ukuran, tumpang, _, total, _, hit, mrr in hasil_semua:
        isi = f"{nama:<16}{total:>7}" + "".join(f"{hit[k]:>9.2f}" for k in NILAI_K) + f"{mrr:>8.2f}"
        print(isi)
        baris.append(isi)
    baris.append("")

    # --- Rincian per koleksi ---
    for nama, ukuran, tumpang, ringkasan, total, rincian, hit, mrr in hasil_semua:
        baris += [garis, f"KOLEKSI {nama} | chunk={ukuran} | overlap={tumpang} | total {total} chunk", garis]
        for nm, n in ringkasan:
            baris.append(f"  {nm} -> {n} chunk")
        baris.append("")
        baris += [f"  Hit@{k} = {hit[k]:.2f}" for k in NILAI_K]
        baris += [f"  MRR   = {mrr:.2f}", "", "  RINCIAN PER PERTANYAAN"]

        gagal = []
        for no, r in enumerate(rincian, start=1):
            if r["peringkat"] is None:
                status = "GAGAL (tidak ada di top-%d)" % max(NILAI_K)
                gagal.append((no, r))
            else:
                status = f"kena di peringkat {r['peringkat']}"
            baris.append(f"  [{no:>2}] {status} | {r['pertanyaan']} | kunci: \"{r['kunci']}\"")

        if gagal:
            baris += ["", "  ANALISIS PERTANYAAN YANG GAGAL"]
            for no, r in gagal:
                baris.append(f"  [{no}] {r['pertanyaan']}  (kunci: \"{r['kunci']}\")")
                if not r["kunci_ada_di_korpus"]:
                    baris.append("      ! Kata kunci TIDAK ditemukan di chunk mana pun pada koleksi ini: "
                                 "cek penulisan kunci, atau kata kunci terpotong di perbatasan chunk.")
                m = r["top1_meta"]
                if m:
                    baris.append(f"      Chunk teratas: [{m['sumber']} · chunk {m['chunk']}] jarak={r['top1_jarak']:.3f}")
                    baris.append("      " + r["top1_doc"][:300].strip().replace("\n", " ") + "...")
        baris.append("")

    # --- Template kesimpulan ---
    baris += [garis, "KESIMPULAN (isi sendiri, 3-4 kalimat)", garis,
              "Ukuran chunk yang lebih baik : ",
              "Pertanyaan yang gagal        : ",
              "Dugaan penyebab              : ", ""]

    FOLDER_HASIL.mkdir(parents=True, exist_ok=True)
    sudah_ada = FILE_HASIL.exists()
    FILE_HASIL.write_text("\n".join(baris) + "\n", encoding="utf-8")
    print(f"\nHasil lengkap {'diperbarui' if sudah_ada else 'dibuat'}: {FILE_HASIL}")
    print("Tulis kesimpulan 3-4 kalimat: ukuran mana lebih baik, pertanyaan apa yang gagal, dugaan penyebabnya.")


if __name__ == "__main__":
    main()