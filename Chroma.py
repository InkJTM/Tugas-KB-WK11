import random
import sys
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
FOLDER_PDF = BASE_DIR / "dokumen"
FOLDER_HASIL = BASE_DIR / "hasil"      # folder untuk hasil Lab 1, 2, 3
FILE_HASIL = FOLDER_HASIL / "Lab1_test.txt"
UKURAN_UJI = (500, 1000, 1500)
PERSEN_OVERLAP = 0.15  # 15% (rentang yang disarankan: 10-20%)
JUMLAH_SAMPEL = 3


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
        if mulai + ukuran >= len(teks):  # sudah sampai akhir teks
            break
        mulai += ukuran - tumpang
    return [h for h in hasil if h.strip()]


def simpan_hasil(pdf, teks, hasil_per_ukuran):
    """Tulis semua chunk ke hasil/Lab1_test.txt.
    Folder dibuat jika belum ada; file lama ditimpa dengan hasil terbaru."""
    FOLDER_HASIL.mkdir(parents=True, exist_ok=True)
    sudah_ada = FILE_HASIL.exists()

    garis = "=" * 70
    with open(FILE_HASIL, "w", encoding="utf-8") as f:
        f.write(f"{garis}\nLAB 1 - HASIL CHUNKING\n{garis}\n")
        f.write(f"Diperbarui : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"File PDF   : {pdf.name}\n")
        f.write(f"Panjang teks: {len(teks)} karakter\n\n")

        f.write("RINGKASAN\n")
        for ukuran, (tumpang, chunks) in hasil_per_ukuran.items():
            f.write(f"  ukuran={ukuran:<5} overlap={tumpang:<4} -> {len(chunks)} chunk\n")

        for ukuran, (tumpang, chunks) in hasil_per_ukuran.items():
            f.write(f"\n{garis}\nUKURAN {ukuran} | OVERLAP {tumpang} | {len(chunks)} CHUNK\n{garis}\n")
            for i, c in enumerate(chunks):
                f.write(f"\n--- chunk {i} ({len(c)} karakter) ---\n{c}\n")

    status = "diperbarui" if sudah_ada else "dibuat"
    print(f"\nHasil lengkap {status}: {FILE_HASIL}")


def pilih_pdf():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            sys.exit(f"File tidak ditemukan: {path}")
        return path

    if not FOLDER_PDF.exists():
        FOLDER_PDF.mkdir(parents=True)
        sys.exit(f"Folder '{FOLDER_PDF}' baru dibuat. Isi dengan PDF, lalu jalankan lagi.")
    pdfs = sorted(FOLDER_PDF.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Tidak ada file .pdf di '{FOLDER_PDF}'. Tambahkan PDF dulu.")
    return pdfs[0]


def main():
    pdf = pilih_pdf()

    # Langkah 1: pastikan teks PDF terbaca dengan baik
    teks = baca_pdf(pdf)
    print("=" * 60)
    print(f"LANGKAH 1 - Cek teks: {pdf.name} ({len(teks)} karakter)")
    print("=" * 60)
    if not teks.strip():
        sys.exit("Teks kosong. PDF ini kemungkinan hasil scan; pakai PDF berbasis teks.")
    print(teks[:500])

    # Langkah 2: potong dengan tiga ukuran, catat jumlah chunk
    print("\n" + "=" * 60)
    print("LANGKAH 2 - Jumlah chunk per pengaturan")
    print("=" * 60)
    hasil_per_ukuran = {}
    for ukuran in UKURAN_UJI:
        tumpang = int(ukuran * PERSEN_OVERLAP)
        chunks = potong(teks, ukuran, tumpang)
        hasil_per_ukuran[ukuran] = (tumpang, chunks)
        print(f"ukuran={ukuran:<5} overlap={tumpang:<4} -> {len(chunks)} chunk")

    # Langkah 3: lihat chunk acak, adakah yang terpotong di tengah kalimat/tabel?
    print("\n" + "=" * 60)
    print(f"LANGKAH 3 - {JUMLAH_SAMPEL} chunk acak per ukuran")
    print("=" * 60)
    for ukuran, (tumpang, chunks) in hasil_per_ukuran.items():
        print(f"\n##### ukuran={ukuran}, overlap={tumpang} ({len(chunks)} chunk) #####")
        for c in random.sample(chunks, min(JUMLAH_SAMPEL, len(chunks))):
            print("-" * 40)
            print(f"[awal]  {c[:150].strip()}")
            print(f"[akhir] ...{c[-150:].strip()}")

    simpan_hasil(pdf, teks, hasil_per_ukuran)

    print("\nCatat: jumlah chunk tiap ukuran, dan apakah ada chunk yang")
    print("terpotong di tengah kalimat atau tabel (lihat bagian [awal]/[akhir]).")


if __name__ == "__main__":
    main()