# Sistem Rekomendasi Film

**Penulis:** Radithya Fawwaz Aydin

Sistem rekomendasi film berbasis konten menggunakan dataset The Movie Database (TMDb) yang memberikan saran film berdasarkan analisis kesamaan konten.

## 🎯 Gambaran Proyek

Industri perfilman memproduksi ribuan film setiap tahunnya, sehingga penonton mengalami kesulitan untuk menemukan konten yang sesuai dengan preferensi mereka. Proyek ini mengembangkan sistem rekomendasi film menggunakan **content-based filtering** untuk membantu pengguna menemukan film yang relevan berdasarkan karakteristik film seperti genre, kata kunci, pemeran, kru, dan sinopsis.

### Mengapa Masalah Ini Penting
- **Kelebihan Informasi**: Dengan ribuan pilihan film, penonton sulit memilih konten yang sesuai
- **Masalah Penemuan**: Pengguna kesulitan menemukan film baru yang sesuai selera mereka
- **Nilai Bisnis**: Platform streaming memerlukan sistem rekomendasi yang efektif untuk meningkatkan engagement dan retensi pengguna

## 🎬 Pemahaman Bisnis

### Pernyataan Masalah
1. **Bagaimana memberikan rekomendasi film yang akurat berdasarkan karakteristik konten film?**
   - Pengguna seringkali menyukai film dengan genre, tema, atau gaya serupa
   - Diperlukan sistem yang dapat mengidentifikasi kesamaan antar film

2. **Bagaimana mengatasi cold start problem dalam sistem rekomendasi?**
   - Film baru atau pengguna baru tidak memiliki riwayat interaksi
   - Sistem harus dapat memberikan rekomendasi tanpa bergantung pada data historis pengguna

3. **Bagaimana mengukur efektivitas sistem rekomendasi yang dibangun?**
   - Diperlukan metrik evaluasi yang tepat untuk mengukur kualitas rekomendasi
   - Sistem harus memberikan hasil yang konsisten dan relevan

### Tujuan
1. **Mengembangkan sistem rekomendasi film berbasis konten** yang memberikan 5 rekomendasi teratas dengan akurasi tinggi
2. **Membangun sistem yang robust** yang dapat menangani cold start problem dengan efektif
3. **Melakukan evaluasi komprehensif** terhadap kualitas rekomendasi

### Pendekatan Solusi
- **Content-Based Filtering dengan TF-IDF Vectorization**
- **Text Processing dan Feature Engineering**
- **Cosine Similarity untuk perhitungan skor rekomendasi**

## 📊 Dataset

**Sumber:** [TMDb Movie Dataset di Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

### Struktur Dataset
- **tmdb_5000_movies.csv**: 4.803 film × 20 fitur
- **tmdb_5000_credits.csv**: 4.803 film × 4 fitur

### Fitur Utama yang Digunakan
- **Fitur Konten**: Overview, genre, kata kunci
- **Orang**: Pemeran (3 aktor utama), kru (sutradara)
- **Metadata**: Judul, ID film

## 🔧 Persiapan Data

### 1. Penggabungan Data
```python
movies = movies.merge(credits, on='title')
```

### 2. Seleksi Fitur
Memilih fitur yang relevan untuk rekomendasi berbasis konten:
```python
movies_fix = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```

### 3. Pembersihan Data
- Menghapus nilai yang hilang pada kolom overview
- Memeriksa dan menangani data duplikat

### 4. Feature Engineering
- **Parsing JSON**: Mengekstrak nama dari kolom berformat JSON
- **Preprocessing Teks**: Tokenisasi dan stemming
- **Kombinasi Fitur**: Menggabungkan semua fitur menjadi tags terpadu
- **Normalisasi**: Menerapkan Porter Stemming untuk konsistensi teks

### 5. Vektorisasi
```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

## 🤖 Pemodelan

### Algoritma: Content-Based Filtering dengan Cosine Similarity

**Formula Cosine Similarity:**
```
similarity(A,B) = cos(θ) = (A·B) / (||A|| × ||B||)
```

### Implementasi
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(judul_film):
    indeks_film = new_df[new_df['title'] == judul_film].index[0]
    jarak = similarity[indeks_film]
    daftar_film = sorted(list(enumerate(jarak)), reverse=True, key=lambda x: x[1])[1:6]
    
    rekomendasi = []
    for i in daftar_film:
        rekomendasi.append(new_df.iloc[i[0]].title)
    return rekomendasi
```

### Mengapa Cosine Similarity?
- **Tidak Bergantung Magnitude**: Tidak terpengaruh panjang dokumen
- **Berbasis Sudut**: Fokus pada arah vektor daripada besaran
- **Efektif**: Bekerja baik dengan data sparse dimensi tinggi

## 📈 Hasil & Evaluasi

### Kasus Uji

#### Kasus Uji 1: "Superman Returns"
**Rekomendasi:**
1. Superman II ✅
2. Superman III ✅
3. Superman IV: The Quest for Peace ✅
4. Superman ✅
5. The Wolverine ⚠️

**Presisi:** 80% (4/5 rekomendasi relevan)

#### Kasus Uji 2: "Tangled"
**Rekomendasi:**
1. Out of Inferno ❓
2. The Princess and the Frog ✅
3. Home on the Range ✅
4. Animals United ✅
5. Toy Story 3 ✅

**Presisi:** 80% (4/5 rekomendasi relevan)

### Analisis Distribusi Similarity
```
Rata-rata Similarity: 0.053
Standar Deviasi: 0.056
Rentang: 0.000 - 1.000
```

**Interpretasi:**
- Rata-rata similarity rendah menunjukkan keragaman konten yang baik
- Standar deviasi moderat menunjukkan distribusi yang sehat
- Rentang penuh menunjukkan kemampuan sistem membedakan berbagai tingkat kesamaan

## ✅ Performa Model

### Kekuatan
- **Tidak Ada Cold Start Problem**: Bekerja untuk film baru tanpa riwayat interaksi
- **Transparansi**: Rekomendasi dapat dijelaskan
- **Konsistensi**: Input sama menghasilkan output sama
- **Tingkat Presisi**: 80% berdasarkan evaluasi manual

### Keterbatasan
- **Keragaman Terbatas**: Cenderung merekomendasikan film yang sangat mirip
- **Bergantung Konten**: Hanya mengandalkan fitur konten yang dapat diekstrak
- **Over-spesialisasi**: Dapat menciptakan "filter bubble"

## 🚀 Perbaikan Masa Depan

1. **Pendekatan Hibrid**: Integrasikan collaborative filtering untuk keragaman yang lebih baik
2. **Fitur Berbobot**: Berikan bobot berbeda untuk berbagai komponen konten
3. **Metrik Lanjutan**: Implementasikan NDCG, MAP untuk evaluasi komprehensif
4. **Optimisasi Parameter**: Fine-tune ukuran vocabulary dan parameter preprocessing

## 📋 Persyaratan

```bash
pandas
numpy
scikit-learn
nltk
ast
```

## 🏃‍♂️ Cara Menjalankan

1. **Instal dependensi**
```bash
pip install pandas numpy scikit-learn nltk
```

2. **Unduh dataset** dari [Kaggle TMDb Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

3. **Jalankan sistem rekomendasi**
```python
# Muat dan persiapkan data
# Latih model
# Dapatkan rekomendasi
rekomendasi = recommend("Judul Film Anda")
print(rekomendasi)
```

## 📚 Referensi

[1] F. O. Isinkaye, Y. O. Folajimi, and B. A. Ojokoh, "Recommendation systems: Principles, methods and evaluation," *Egyptian informatics journal*, vol. 16, no. 3, pp. 261-273, 2015.

[2] J. Lu, D. Wu, M. Mao, W. Wang, and G. Zhang, "Recommender system application developments: a survey," *Decision support systems*, vol. 74, pp. 12-32, 2015.

---

**Catatan:** Sistem ini berfokus pada rekomendasi berbasis konten dan berfungsi sebagai fondasi untuk sistem rekomendasi hibrid yang lebih canggih.
