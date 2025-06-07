# Laporan Proyek Machine Learning - Sistem Rekomendasi Film (Radithya Fawwaz Aydin)

## Project Overview

Industri perfilman mengalami perkembangan pesat dengan ribuan film diproduksi setiap tahunnya. Dengan jumlah konten yang sangat besar, penonton seringkali kesulitan menemukan film yang sesuai dengan preferensi mereka. Sistem rekomendasi menjadi solusi untuk membantu penonton menemukan film yang menarik berdasarkan karakteristik konten film.

Proyek ini mengembangkan sistem rekomendasi film menggunakan pendekatan content-based filtering dengan memanfaatkan data dari The Movie Database (TMDb). Sistem ini akan menganalisis karakteristik film seperti genre, kata kunci, cast, crew, dan sinopsis untuk memberikan rekomendasi film yang serupa.

**Mengapa masalah ini penting untuk diselesaikan:**
- **Information Overload**: Dengan ribuan pilihan film, penonton mengalami kesulitan dalam memilih konten yang sesuai
- **Discovery Problem**: Penonton sulit menemukan film baru yang sesuai dengan selera mereka
- **Business Value**: Platform streaming memerlukan sistem rekomendasi yang efektif untuk meningkatkan engagement dan retention pengguna

**Referensi:**
[1] F. O. Isinkaye, Y. O. Folajimi, and B. A. Ojokoh, "Recommendation systems: Principles, methods and evaluation," *Egyptian informatics journal*, vol. 16, no. 3, pp. 261-273, 2015.

[2] J. Lu, D. Wu, M. Mao, W. Wang, and G. Zhang, "Recommender system application developments: a survey," *Decision support systems*, vol. 74, pp. 12-32, 2015.

## Business Understanding

Dalam era digital saat ini, platform streaming dan database film memerlukan sistem yang dapat membantu pengguna menemukan konten yang relevan dengan cepat dan akurat. Proyek ini bertujuan untuk mengembangkan solusi yang dapat meningkatkan pengalaman pengguna dalam pencarian film.

### Problem Statements

1. **Bagaimana cara memberikan rekomendasi film yang akurat berdasarkan karakteristik konten film?**
   - Pengguna seringkali menyukai film dengan genre, tema, atau gaya serupa
   - Diperlukan sistem yang dapat mengidentifikasi kesamaan antar film

2. **Bagaimana cara mengatasi cold start problem dalam sistem rekomendasi?**
   - Film baru atau pengguna baru tidak memiliki riwayat interaksi
   - Sistem harus dapat memberikan rekomendasi tanpa bergantung pada data historis pengguna

3. **Bagaimana cara mengukur efektivitas sistem rekomendasi yang dibangun?**
   - Diperlukan metrik evaluasi yang tepat untuk mengukur kualitas rekomendasi
   - Sistem harus dapat memberikan hasil yang konsisten dan relevan

### Goals

1. **Mengembangkan sistem rekomendasi film berbasis konten yang dapat memberikan 5 rekomendasi teratas dengan akurasi tinggi**
   - Memanfaatkan fitur seperti genre, cast, crew, keywords, dan overview
   - Menghasilkan rekomendasi yang relevan dengan film input

2. **Membangun sistem yang robust dan dapat menangani cold start problem**
   - Sistem tidak memerlukan data interaksi pengguna
   - Dapat memberikan rekomendasi untuk film baru berdasarkan konten

3. **Melakukan evaluasi komprehensif terhadap kualitas rekomendasi yang dihasilkan**
   - Menganalisis distribusi similarity scores
   - Melakukan pengujian kualitatif dan kuantitatif

### Solution Statements

1. **Content-Based Filtering dengan TF-IDF Vectorization**
   - Menggunakan CountVectorizer untuk mengkonversi teks menjadi vektor numerik
   - Menggabungkan berbagai fitur teks (overview, genres, keywords, cast, crew) menjadi satu representasi
   - Menghitung similarity menggunakan cosine similarity

2. **Text Processing dan Feature Engineering**
   - Preprocessing teks dengan tokenization dan stemming
   - Ekstraksi informasi dari data JSON (cast, crew, genres)
   - Normalisasi dan cleaning data untuk meningkatkan kualitas fitur

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari The Movie Database (TMDb) yang terdiri dari dua file CSV utama dengan total 5000 film. Dataset ini dapat diakses melalui [Kaggle TMDb Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

**Informasi Dataset:**
1. tmdb_5000_movies.csv
- **Jumlah Data**: 4803 baris dan 20 kolom,
- **Sumber**: The Movie Database (TMDb)
- **Format**: file CSV
- **Kondisi**: Data berkualitas baik dengan minimal missing values
2. tmdb_5000_credits.csv
- **Jumlah Data**: 4803 film dan 4 kolom
- **Sumber**: The Movie Database (TMDb)
- **Format**: file CSV
- **Kondisi**: Data berkualitas baik dan lengkap (no missing values)


### Struktur Dataset

**File 1: tmdb_5000_movies.csv**
- **budget**: Anggaran produksi film (numerik)
- **genres**: Genre film dalam format JSON (string)
- **homepage**: Website resmi film (string)
- **id**: ID unik film (numerik)
- **keywords**: Kata kunci terkait film dalam format JSON (string)
- **original_language**: Bahasa asli film (string)
- **original_title**: Judul asli film (string)
- **overview**: Sinopsis film (text)
- **popularity**: Skor popularitas film (numerik)
- **production_companies**: Perusahaan produksi dalam format JSON (string)
- **production_countries**: Negara produksi dalam format JSON (string)
- **release_date**: Tanggal rilis (date)
- **revenue**: Pendapatan film (numerik)
- **runtime**: Durasi film dalam menit (numerik)
- **spoken_languages**: Bahasa yang digunakan dalam format JSON (string)
- **status**: Status rilis film (string)
- **tagline**: Tagline film (string)
- **title**: Judul film (string)
- **vote_average**: Rating rata-rata (numerik)
- **vote_count**: Jumlah voter (numerik)

**File 2: tmdb_5000_credits.csv**
- **movie_id**: ID film yang sesuai dengan file movies (numerik)
- **title**: Judul film (string)
- **cast**: Data pemeran dalam format JSON (string)
- **crew**: Data kru produksi dalam format JSON (string)

### Exploratory Data Analysis

Setelah melakukan penggabungan dataset, ditemukan beberapa insight penting:

1. **Missing Values**: Terdapat 3 data null pada kolom overview yang akan dihapus
2. **Data Format**: Kolom genres, keywords, cast, dan crew dalam format JSON string yang perlu di-parse
3. **Data Quality**: Tidak terdapat data duplikat
4. **Feature Richness**: Dataset memiliki informasi lengkap tentang cast, crew, genre, dan deskripsi film

## Data Preparation

Tahap data preparation merupakan langkah krusial untuk mempersiapkan data agar siap digunakan dalam model sistem rekomendasi. Berikut adalah tahapan yang dilakukan:

### 1. Merging Dataset
***🔗 Data Merging Process: ***
    ```movies = movies.merge(credits, on='title')```

Penjelasan Merge:
- **Join Key**: Menggunakan kolom title sebagai kunci penggabungan
- **Tipe Join**: Default inner join (hanya film yang ada di kedua dataset)
- **Hasil**: Dataset gabungan yang menggabungkan informasi film dengan data cast/crew
- **Jumlah**: 4809 data


### 2. Feature Selection
```python
movies_fix = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```
**Alasan**: Memilih fitur yang relevan untuk content-based recommendation, fokus pada karakteristik konten film yang dapat dianalisis secara tekstual.

### 3. Data Cleaning
```python
# Menghilangkan missing values
movies_fix.dropna(inplace=True)

# Mengecek duplikasi
movies_fix.duplicated().sum()
```
**Alasan**: Missing values pada overview akan mengurangi kualitas representasi film, sehingga data tersebut dihapus. Duplikasi data dapat menyebabkan bias dalam perhitungan similarity.

### 4. JSON Parsing dan Feature Extraction
```python
def convert(object):
    L = []
    for i in ast.literal_eval(object):
        L.append(i['name'])
    return L

# Untuk cast (hanya 3 aktor utama)
def convert3(object):
    L = []
    counter = 0
    for i in ast.literal_eval(object):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

# Untuk crew (hanya director)
def fetch_director(object):
    L = []
    for i in ast.literal_eval(object):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
```
**Alasan**: Data JSON perlu di-parse untuk mengekstrak informasi yang relevan. Pembatasan cast pada 3 aktor utama untuk menghindari noise, sedangkan crew difokuskan pada director karena memiliki pengaruh signifikan terhadap karakteristik film.

### 5. Text Preprocessing
```python
# Tokenisasi overview
movies_fix['overview'] = movies_fix['overview'].apply(lambda x: x.split())

# Menghilangkan spasi dalam nama
movies_fix['genres'] = movies_fix['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
```
**Alasan**: Tokenisasi diperlukan untuk memecah teks menjadi unit-unit kata. Menghilangkan spasi dalam nama mencegah pemisahan nama yang seharusnya menjadi satu entitas.

### 6. Feature Engineering
```python
# Menggabungkan semua fitur menjadi tags
movies_fix['tags'] = movies_fix['overview'] + movies_fix['genres'] + movies_fix['keywords'] + movies_fix['cast'] + movies_fix['crew']

# Menggabungkan menjadi string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
```
**Alasan**: Menggabungkan semua fitur relevan menjadi satu representasi teks yang komprehensif untuk memudahkan proses vectorization.

### 7. Text Normalization dengan Stemming
```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
```
**Alasan**: Stemming mengurangi variasi kata dengan mengubah kata ke bentuk dasarnya (contoh: "running" menjadi "run"), sehingga meningkatkan akurasi perhitungan similarity.

### 8. Count Vectorizer
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

**Penjelasan Parameter:**
- **max_features=5000**: Mengambil 5000 kata yang paling sering muncul untuk mengurangi dimensi dan noise
- **stop_words='english'**: Menghilangkan kata-kata umum seperti "the", "and", "is" yang tidak memberikan informasi discriminative
- **Output**: Matrix berukuran 4806 x 5000 (jumlah film x vocabulary size)

## Modeling

Sistem rekomendasi yang dikembangkan menggunakan pendekatan content-based filtering dengan algoritma cosine similarity. Berikut adalah tahapan modeling yang dilakukan:


### 1. Similarity Calculation
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

**Cosine Similarity Formula:**
```
similarity(A,B) = cos(θ) = (A·B) / (||A|| × ||B||)
```

**Keunggulan cosine similarity:**
- Tidak terpengaruh oleh panjang dokumen
- Fokus pada sudut antar vektor, bukan magnitude
- Efektif untuk high-dimensional sparse data

### 2. Recommendation Function
```python
def recommend(movies):
    # Mencari index film input
    movies_index = new_df[new_df['title'] == movies].index[0]
    
    # Mengambil similarity scores
    distances = similarity[movies_index]
    
    # Sorting berdasarkan similarity (descending)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # Menampilkan top 5 rekomendasi
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

##### Test Case 1: "Superman Returns"
**Rekomendasi yang dihasilkan:**
1. Superman II 
2. Superman III 
3. Superman IV: The Quest for Peace 
4. Superman 
5. The Wolverine 

##### Test Case 2: "Tangled"
**Rekomendasi yang dihasilkan:**
1. Out of Inferno 
2. The Princess and the Frog 
3. Home on the Range 
4. Animals United 
5. Toy Story 3 

### Kelebihan dan Kekurangan Pendekatan

**Kelebihan Content-Based Filtering:**
- **No Cold Start Problem**: Dapat memberikan rekomendasi untuk film baru tanpa memerlukan data interaksi
- **Transparency**: Dapat dijelaskan mengapa film tertentu direkomendasikan
- **Independence**: Tidak memerlukan data pengguna lain
- **Consistency**: Memberikan hasil yang konsisten untuk input yang sama

**Kekurangan:**
- **Limited Diversity**: Cenderung merekomendasikan film yang sangat mirip
- **Content Limitation**: Terbatas pada fitur yang dapat diekstrak dari konten
- **New User Problem**: Tidak dapat mempelajari preferensi individual pengguna
- **Over-specialization**: Mungkin terjebak dalam "filter bubble"

## Evaluation

Evaluasi sistem rekomendasi dilakukan melalui dua pendekatan: analisis kualitatif dan kuantitatif untuk mengukur kualitas rekomendasi yang dihasilkan.

### Metrik Evaluasi

#### 1. Precision Rate

##### Test Case 1: "Superman Returns"
**Rekomendasi yang dihasilkan:**
1. Superman II ✅
2. Superman III ✅
3. Superman IV: The Quest for Peace ✅
4. Superman ✅
5. The Wolverine ⚠️

**Analisis**: Precision = 4/5 (80%) - Sangat baik, mayoritas adalah film dari franchise Superman yang sama.

##### Test Case 2: "Tangled"
**Rekomendasi yang dihasilkan:**
1. Out of Inferno ❓
2. The Princess and the Frog ✅
3. Home on the Range ✅
4. Animals United ✅
5. Toy Story 3 ✅

**Analisis**: Precision = 4/5 (80%) - Baik, mayoritas adalah film animasi/family yang sesuai dengan genre Tangled.

#### 2. Similarity Distribution Analysis
```python
def analyze_recommendation_quality(similarity_scores):
    return {
        'mean_similarity': np.mean(similarity_scores),
        'std_similarity': np.std(similarity_scores),
        'min_similarity': np.min(similarity_scores),
        'max_similarity': np.max(similarity_scores)
    }
```
##### Distribusi Cosine Similarity
```python
{
    'mean_similarity': 0.053,
    'std_similarity': 0.056,
    'min_similarity': 0.000,
    'max_similarity': 1.000
}
```

**Interpretasi Metrik:**

1. **Mean Similarity (0.053)**: 
   - Nilai rata-rata similarity yang rendah menunjukkan bahwa sebagian besar film memiliki perbedaan yang signifikan
   - Hal ini positif karena menunjukkan dataset memiliki keragaman konten yang baik
   - Sistem tidak akan memberikan rekomendasi yang terlalu umum

2. **Standard Deviation (0.056)**:
   - Variasi similarity yang moderat menunjukkan distribusi yang sehat
   - Tidak terlalu clustered maupun terlalu tersebar

3. **Range (0.000 - 1.000)**:
   - Rentang penuh dari tidak mirip sama sekali hingga identik sempurna
   - Menunjukkan sistem dapat membedakan film dengan berbagai tingkat kesamaan

### Kesimpulan Evaluasi

**Performa Sistem:**
- **Precision Rate**: 80% berdasarkan manual evaluation
- **Consistency**: Tinggi - memberikan hasil yang sama untuk input yang sama
- **Relevance**: Baik - rekomendasi sesuai dengan genre dan karakteristik film input
- **Diversity**: Sedang - masih dalam koridor genre yang sama tetapi dengan variasi

**Kekuatan Model:**
1. Efektif untuk film dengan franchise atau genre yang jelas
2. Tidak memerlukan data historis pengguna
3. Dapat menjelaskan alasan rekomendasi
4. Cepat dalam memberikan hasil

**Area Perbaikan:**
1. Integrasi dengan collaborative filtering untuk meningkatkan diversity
2. Penambahan weighted features untuk memberikan importance yang berbeda pada setiap komponen
3. Implementasi evaluation metrics yang lebih comprehensive seperti NDCG atau MAP
4. Optimisasi vocabulary size dan preprocessing parameters

Sistem rekomendasi yang dikembangkan telah berhasil mencapai tujuan awal dengan memberikan rekomendasi yang relevan dan akurat berdasarkan konten film. Dengan precision rate 80%, sistem ini dapat menjadi foundation yang solid untuk pengembangan lebih lanjut ke arah hybrid recommendation system.#   m o v i e s _ r e c s y s _ t m d b 5 0 0 0  
 