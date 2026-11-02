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
FILE_HASIL = FOLDER_HASIL / "Lab2_test.txt"

MODEL_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
NAMA_KOLEKSI = "pedoman_800"
UKURAN_CHUNK = 800
OVERLAP_CHUNK = 120   # ±15%
TOP_K = 3

# 5 pertanyaan biasa (GANTI sesuai isi PDF Anda)
PERTANYAAN_BIASA = [
"Mengapa masyarakat perlu tetap berpikir kritis dan tidak membela pemimpin secara membabi buta, meskipun mereka merupakan pendukungnya?",

"Bagaimana pengaruh buzzer politik yang menyebarkan narasi atau informasi menyesatkan terhadap kualitas demokrasi dan kebebasan berpendapat di Indonesia?",

"Apa saja hal yang seharusnya menjadi pertimbangan masyarakat sebelum menentukan pilihan dalam pemilu, dan mengapa rekam jejak calon pemimpin penting untuk diperhatikan?",

"Mengapa kritik dari masyarakat merupakan bagian penting dalam mengevaluasi kinerja seorang presiden, bukan bentuk kebencian atau ketidaksetiaan terhadap negara?",

"Bagaimana cara membedakan pemimpin yang benar-benar memiliki kemampuan memimpin dengan pemimpin yang hanya mengandalkan popularitas, janji, dan pencitraan politik?"
]
# 2 pertanyaan parafrase: kata-katanya BERBEDA dari teks dokumen
PERTANYAAN_PARAFRASE = [
    "Bolehkah berhenti kuliah sementara waktu?",
    "parafrase pertanyaan lain ...",
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


def bangun_indeks(emb, log):
    """Buat koleksi baru dari semua PDF. Koleksi lama dengan nama sama dihapus
    dulu agar tidak ada sisa data. Mengembalikan (koleksi, ringkasan per PDF)."""
    import chromadb

    client = chromadb.PersistentClient(path=str(FOLDER_DB))
    try:
        client.delete_collection(NAMA_KOLEKSI)
    except Exception:
        pass
    koleksi = client.create_collection(
        name=NAMA_KOLEKSI, embedding_function=emb,
        metadata={"hnsw:space": "cosine"})  # jarak cosine

    ringkasan = []
    for pdf in daftar_pdf():
        chunks = potong(baca_pdf(pdf), UKURAN_CHUNK, OVERLAP_CHUNK)
        if not chunks:
            log(f"{pdf.name} -> 0 chunk (PDF scan / teks kosong?) DILEWATI")
            ringkasan.append((pdf.name, 0))
            continue
        koleksi.upsert(
            ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))],
        )
        log(f"{pdf.name} -> {len(chunks)} chunk")
        ringkasan.append((pdf.name, len(chunks)))
    return koleksi, ringkasan


def cari(koleksi, pertanyaan, k=TOP_K):
    hasil = koleksi.query(query_texts=[pertanyaan], n_results=k)
    return list(zip(hasil["documents"][0],
                    hasil["metadatas"][0],
                    hasil["distances"][0]))


def pertanyaan_valid(daftar):
    """Lewati placeholder yang belum diganti (berakhiran '...')."""
    valid = [q for q in daftar if not q.strip().endswith("...")]
    lewat = len(daftar) - len(valid)
    if lewat:
        print(f"PERINGATAN: {lewat} pertanyaan masih placeholder ('...') dan dilewati. "
              "Ganti di bagian atas file.")
    return valid


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    garis = "=" * 70
    baris_file = []   # semua yang ditulis ke Lab2_test.txt

    def log(teks=""):
        print(teks)

    biasa = pertanyaan_valid(PERTANYAAN_BIASA)
    parafrase = pertanyaan_valid(PERTANYAAN_PARAFRASE)
    if not biasa and not parafrase:
        sys.exit("Tidak ada pertanyaan yang valid. Isi PERTANYAAN_BIASA / PERTANYAAN_PARAFRASE.")

    emb = buat_embedding()

    print(garis)
    print("LAB 2 - Bangun indeks")
    print(garis)
    koleksi, ringkasan = bangun_indeks(emb, log)
    total = koleksi.count()
    print(f"Total chunk di koleksi '{NAMA_KOLEKSI}': {total}")

    baris_file += [garis, "LAB 2 - BANGUN INDEKS & CARI", garis,
                   f"Diperbarui : {datetime.now():%Y-%m-%d %H:%M:%S}",
                   f"Model      : {MODEL_EMBEDDING}",
                   f"Koleksi    : {NAMA_KOLEKSI} (chunk={UKURAN_CHUNK}, overlap={OVERLAP_CHUNK}, jarak=cosine)",
                   f"Top-k      : {TOP_K}", "",
                   "INDEKS"]
    for nama, n in ringkasan:
        baris_file.append(f"  {nama} -> {n} chunk")
    baris_file += [f"  Total: {total} chunk", ""]

    def jalankan_set(judul, daftar):
        print("\n" + garis)
        print(judul)
        print(garis)
        baris_file.extend(["", garis, judul, garis])
        for no, q in enumerate(daftar, start=1):
            hasil = cari(koleksi, q)
            print(f"\n[{no}] PERTANYAAN: {q}")
            baris_file.extend(["", f"[{no}] PERTANYAAN: {q}"])
            for rank, (doc, meta, jarak) in enumerate(hasil, start=1):
                kepala = f"#{rank} [{meta['sumber']} · chunk {meta['chunk']}]  jarak={jarak:.3f}"
                print("  " + kepala)
                print("     " + doc[:200].replace("\n", " ").strip() + "...")
                baris_file.extend(["  " + kepala,
                                   "  " + doc.strip().replace("\n", "\n  "), ""])
            baris_file.append("  Chunk teratas relevan?  [ ] ya   [ ] tidak   Catatan: ")

    if biasa:
        jalankan_set(f"{len(biasa)} PERTANYAAN BIASA", biasa)
    if parafrase:
        jalankan_set(f"{len(parafrase)} PERTANYAAN PARAFRASE (kata berbeda dari dokumen)", parafrase)

    FOLDER_HASIL.mkdir(parents=True, exist_ok=True)
    sudah_ada = FILE_HASIL.exists()
    FILE_HASIL.write_text("\n".join(baris_file) + "\n", encoding="utf-8")
    print(f"\nHasil lengkap {'diperbarui' if sudah_ada else 'dibuat'}: {FILE_HASIL}")
    print("Periksa tiap pertanyaan: apakah chunk teratas benar-benar relevan?")
    print("Untuk parafrase: apakah pencarian semantik tetap menemukannya?")


if __name__ == "__main__":
    main()