import os
import shutil

def perbanyak_video_custom(file_sumber, folder_tujuan, jumlah_salinan, nama_baru_custom=None):
    # Memeriksa apakah file sumber ada
    if not os.path.exists(file_sumber):
        print(f"Error: File '{file_sumber}' tidak ditemukan. Pastikan nama file dan lokasinya benar.")
        return

    # Membuat folder tujuan jika belum ada
    if not os.path.exists(folder_tujuan):
        os.makedirs(folder_tujuan)
        print(f"Folder '{folder_tujuan}' berhasil dibuat.")

    # Mengambil ekstensi asli file (misal: .mp4, .mkv)
    _, ekstensi = os.path.splitext(file_sumber)
    
    # Mengambil nama file asli sebagai cadangan jika nama_baru_custom tidak diisi
    nama_asli, _ = os.path.splitext(os.path.basename(file_sumber))

    # Tentukan nama dasar yang akan digunakan
    if nama_baru_custom:
        nama_dasar = nama_baru_custom
    else:
        nama_dasar = nama_asli

    print(f"Mulai memperbanyak video menjadi '{nama_dasar}_[nomor]{ekstensi}' sebanyak {jumlah_salinan} kali...")

    # Proses duplikasi
    for i in range(1, jumlah_salinan + 1):
        # Membuat nama file baru sesuai custom (Contoh: tidak_1.mp4, tidak_2.mp4)
        nama_file_baru = f"{nama_dasar}_{i}{ekstensi}"
        path_tujuan = os.path.join(folder_tujuan, nama_file_baru)

        # Menyalin file
        try:
            shutil.copy2(file_sumber, path_tujuan)
            print(f"[{i}/{jumlah_salinan}] Berhasil membuat: {nama_file_baru}")
        except Exception as e:
            print(f"Gagal menyalin file ke-{i}. Error: {e}")

    print("\nSelesai! Semua file video telah berhasil diperbanyak.")

# ==========================================
# KONFIGURASI PROGRAM
# ==========================================
if __name__ == "__main__":
    # 1. Lokasi file video asli yang ingin diperbanyak
    # Ganti "video_asli.mp4" dengan nama file video yang Anda miliki
    FILE_ASLI = "WhatsApp Video 2026-06-10 at 13.07.50.mp4" 
    
    # 2. Nama folder tempat hasil salinan akan disimpan
    FOLDER_HASIL = "sudahdiperbanyak/hasil" 
    
    # 3. Jumlah salinan yang diinginkan (1 sampai 10)
    JUMLAH_COPY = 10 

    # 4. CUSTOM NAMA DI SINI
    # Akan menghasilkan format: tidak_1, tidak_2, ..., tidak_10
    NAMA_KUSTOM = "Bantu_21"

    # Menjalankan fungsi utama
    perbanyak_video_custom(FILE_ASLI, FOLDER_HASIL, JUMLAH_COPY, NAMA_KUSTOM)