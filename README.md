# Laporan Proyek Machine Learning - Ryan Ananta

## Domain Proyek

Asuransi kesehatan adalah sebuah jenis produk asuransi yang secara khusus menjamin biaya kesehatan atau perawatan para anggota asuransi tersebut jika mereka jatuh sakit atau mengalami kecelakaan.
Untuk mengembangkan produk asuransi kesehatan terbaik yang diminati oleh masyarakat, perusahaan asuransi harus memiliki akses ke data historis untuk memperkirakan biaya medis setiap pengguna. 
Perusahaan asuransi medis dapat menggunakan data untuk mengembangkan model penetapan harga yang lebih akurat.Tujuan dari semua kasus ini adalah untuk memprediksi biaya asuransi secara akurat.

## Business Understanding
### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Fitur apa yang paling berpengaruh terhadap biaya asuransi kesehatan?
- Berapa biaya asuransi kesehatan dengan karakteristik  tertentu?
- Algoritma apa yang dapat memprediksi biaya asuransi kesehatan dengan akurasi terbaik?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui fitur yang berkorelasi tinggi dengan biaya asuransi kesehatan
- Membuat model machine learning yang dapat memprediksi biaya asuransi kesehatan seakurat mungkin berdasarkan fitur-fitur yang ada
- Mengetahui algoritma yang berakurasi tinggi dalam memprediksi biaya asuransi kesehatan.

### Solution statements
- Analisis data dilakukan lebih detail dengan membersihkan data dan beberapa visualisasi data sebelum mencari nilai korelasi
- Melakukan perbandingan terhadap algoritma machine learning yang telah dioptimasi dengan hyperparemeter tuning (Grid Search), yaitu K-Nearest Neighbor (KNN), Random Forest (RF) dan SVM dengan metrik Mean Squared Error (MSE) untuk memperoleh model terbaik

## Data Understanding
Dataset berisi data pemegang polis asuransi kesehatan yang berasal dari Amerika Serikat dengan karakteristik berbeda beserta biaya asuransi yang dibayar. Dataset berasal dari Medical Cost Personal Datasets, data dapat diunduh pada tautan berikut [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance).

Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance), variabel-variabel pada dataset adalah sebagai berikut:

* age: usia peserta asuransi kesehatan
* sex: jenis kelamin peserta asuransi kesehatan
* bmi: indeks massa tubuh peserta asuransi kesehatan
* children: banyaknya anak yang ditanggung oleh asuransi kesehatan
* smoker: peserta asuransi kesehatan merokok atau tidak
* region: domisili peserta asuransi kesehatan (northeast, southeast, southwest, northwest)
* charges: premi asuransi kesehatan yang dibayarkan

Pada bagian ini dilakukan beberapa proses analisis untuk melihat bagaimana kondisi dari dataset. Hal pertama yang dilakukan adalah melihat apakah ada nilai yang kosong pada data dengan cara cek null data dan cek data nol (0) pada fitur bmi dan age, alhasil tidak ada nilai yang kosong pada data.

Kemudian pencarian outlier dilakukan dengan melihat distribusi data menggunakan boxplot pada data numerik seperti berikut.

<img width="264" alt="image" src="https://user-images.githubusercontent.com/58066358/205482186-38b3012f-742e-471d-8f61-a2744ecf38a0.png"> <img width="262" alt="image" src="https://user-images.githubusercontent.com/58066358/205482206-79aaefb6-03a7-42f6-aa97-9156e966b2b0.png"> 
<img width="267" alt="image" src="https://user-images.githubusercontent.com/58066358/205482224-0fd5abab-05e3-4f49-a10d-e644a04eba5b.png"> <img width="264" alt="image" src="https://user-images.githubusercontent.com/58066358/205482229-20300e8a-5fd1-490f-82e6-6ccb056383eb.png">

Secara matematis, pengidentifikasian data outliers dapat dilakukan dengan metode Interquartile Range (IQR). Dalam kasus ini, IQR dimanfaatkan untuk menentukan nilai batas atas dan batas bawah setiap fitur untuk menyaring data-data outliers dalam dataset yang digunakan.

Data yang memiliki satu atau lebih fitur yang berada di luar nilai batas awal dan batas akhir akan dihapus. Setelah penghapusan data outlier, jumlah data berkurang dari 1338 data menjadi 1193 data. Hal ini berarti bahwa terdapat 145 data outlier dalam dataset ini.

Kemudian univariate analysis dilakukan pada satuan data untuk melihat bagaimana persebaran dan jumlah datanya setiap fitur, analisis ini hanya menggunakan beberapa teknik visualisasi yaitu histogram chart dan bar chart.

<img width="290" alt="image" src="https://user-images.githubusercontent.com/58066358/205482367-c2eac09d-7398-463e-92c1-237a265c7fc0.png"> <img width="295" alt="image" src="https://user-images.githubusercontent.com/58066358/205482375-480d294b-147d-406b-b827-2f91fc58409d.png"> <img width="289" alt="image" src="https://user-images.githubusercontent.com/58066358/205482389-93ce38ee-4d94-44a5-86f1-8527984fb572.png">
<img width="465" alt="image" src="https://user-images.githubusercontent.com/58066358/205482409-a47548e4-b978-40d9-9c48-2d17ed435809.png">

Kemudian multivariate analysis dilakukan untuk melihat hubungan setiap fitur variabel dengan fitur target (charges). 
<img width="442" alt="image" src="https://user-images.githubusercontent.com/58066358/205482597-be44deab-5785-441e-b1ea-1b78861a10c4.png">
<img width="441" alt="image" src="https://user-images.githubusercontent.com/58066358/205482659-b6052473-548a-41c9-9416-1dc8676b717a.png">

Berikut correlation matrix yang menilai kolerasi antar fitur numerik.

<img width="427" alt="image" src="https://user-images.githubusercontent.com/58066358/205482695-963aae36-d578-4ee9-8e4e-b6ca1f7f776e.png">

Correlation matrix diatas menunjukkan bahwa terdapat kolerasi yang jelas antara fitur age dan fitur charges. Fitur bmi dan children memiliki kolerasi yang sangat rendah terhadap fitur charges (mendekati 0), Jadi saat ini terdapat lima variabel yang akan digunakan termasuk variabel target yaitu age, sex, smoker, region, dan charges.

## Data Preparation

Setelah analisis dilakukan, pada bagian ini data akan dilakukan beberapa proses teknik data preparation diantaranya yaitu encoding, splitting dataset, dan standardization.

Pada tahapan ini dilakukan proses encoding dengan teknik One-hot Encoding terhadap fitur-fitur kategori. One-hot Encoding merupakan proses pengubahan nilai-nilai pada fitur kategori menjadi format yang dapat diterima oleh model machine learning (berupa numerik). Dalam teknik ini akan dilakukan penambahan variabel atau fitur dummy terhadap dataframe untuk setiap nilai unik dalam fitur kategori. Angka nol dan satu kemudian dimasukkan ke dalam variabel dummy tersebut untuk menunjukkan kategori yang digunakan. Tahapan ini diperlukan untuk meningkatkan akurasi dari model machine learning yang akan dibuat nantinya.

Dataframe setelah mengimplementasi One-hot Encoding.
| | age | charges | Sex_female | Sex_male | Smoker_no | Smoker_yes | Region_northeast | Region_northwest | Region_southeast | Region_southwest |
|---|---|---|---|---|---|---|---|---|---|---|
|0| 19 | 16884.92400 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
|1| 18 | 1725.55230 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
|2| 28 | 4449.46200 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
|3| 33 | 21984.47061 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
|4| 32 | 3866.85520 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
|...| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Setelah itu, splitting dataset dilakukan untuk membagi dataset menjadi data training dan data testing dengan rasio perbandingan 80% data training dan 20% data testing. Tahapan ini dilakukan untuk mempertahankan beberapa data sehingga sebagian data akan dilakukan training pada model kemudian sebagian data lainnya dapat dilakukan testing untuk evaluasi terhadap model yang telah di-training. Sehingga total jumlah data hasil splitting yaitu:

* Total sample in whole dataset: 1193
* Total sample in train dataset: 954
* Total sample in test dataset: 239

Kemudian dilakukan standarisasi terhadap fitur numerik pada `x_train` dan `x_test`. Standarisasi merupakan proses transformasi nilai dari fitur dalam dataset agar nilai-nilai dalam fitur numerik berada pada skala yang relatif sama atau mendekati distribusi normal. Tahapan ini diperlukan agar algoritma yang digunakan dalam model memiliki performa lebih baik dan konvergen lebih cepat.

Dalam kasus ini, standarisasi dilakukan dengan method `StandardScaler()` yang berasal dari [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). StandardScaler berfungsi untuk mengubah nilai-nilai pada fitur numerik, sehingga nilai rata-rata fitur tersebut menjadi 0, dan standar deviasi menjadi 1.

Terdapat satu fitur numerik pada x_train dan x_test, yaitu fitur age. Setelah melakukan standarisasi, fitur age memiliki mean dengan nilai 0 dan standar deviasi dengan nilai 1.


## Modeling
Selanjutnya ketika data sudah siap untuk digunakan maka pengembangan model machine learning akan dilakukan. Proyek ini mengembangkan model machine learning sesuai dengan kasus yang dihadapi yaitu regresi. Terdapat tiga model yang dikembangkan diantaranya K Nearest Neighbors, Random Forest dan Support Vector Machine. Selain mencari model yang terbaik , setiap model juga dilakukan hyperparameter tuning untuk mencari parameter terbaik pada setiap model menggunakan fungsi Grid Search.

K Nearest Neighbors (KNN) adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan 'kesamaan fitur' untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).

Algoritma Random Forest merupakan algoritma yang digunakan dalam kasus klasifikasi dan regresi dengan data dalam jumlah yang besar. Random forest terdiri dari kombinasi dari masing – masing pohon (tree) dari model *Decision Tree*, dan kemudian dikombinasikan ke dalam satu model. Penentuan hasil dilakukan dengan mengambil prediksi terbaik di antara semua model *Decision Tree* yang ada [11].

Algoritma Support Vector Machine merupakan salah satu algoritma yang termasuk dalam kategori Supervised Learning baik kasus klasifikasi maupun regresi, dalam proyek ini SVM digunakan dalam kasus regresi. Tujuan dasar dari algoritma SVM adalah menemukan garis keputusan yang paling sesuai. Dalam SVM, garis yang paling cocok adalah hyperplane yang memiliki jumlah poin maksimum. Tidak seperti model regresi lain yang mencoba meminimalkan kesalahan antara nilai aktual dan nilai prediksi, SVM mencoba menyesuaikan garis terbaik dalam nilai ambang batas (jarak antara hyperplane dan boundary line).

## Evaluation
Tahapan terakhir yang perlu dilakukan adalah evaluasi model machine learning. Seperti yang sudah dijelaskan pada bagian-bagian sebelumnya bahwa proyek ini akan menghitung evaluasi model menggunakan mean squared error (mse). Mean Squared Error adalah Rata-rata Kesalahan kuadrat diantara nilai aktual dan nilai peramalan. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Berikut ini cara menghitung nilai mse dengan n sebagai jumlah data.

Hasil evaluasi dengan metrik MSE

<img width="193" alt="image" src="https://user-images.githubusercontent.com/58066358/205483383-b4c9eac3-c6d2-4c0a-be98-445fa78b0896.png">
<img width="299" alt="image" src="https://user-images.githubusercontent.com/58066358/205483412-2152bd55-b75a-43e0-9e09-b701d122b3fc.png">

Gambar diatas menunjukan bahwa model dengan algoritma Random Forest memiliki nilai MSE terendah dalam pengujian prediksi data testing.

Contoh prediksi

<img width="335" alt="image" src="https://user-images.githubusercontent.com/58066358/205483453-39e78731-c7d6-41a7-9284-41b23b3ddfb9.png">

## Kesimpulan
- Fitur-fitur yang berpengaruh terhadap biaya asuransi kesehatan (*charges*) adalah umur (*age*), status merokok (*smoker*), domisili (*region*), dan jenis kelamin (*sex*).
- Model machine learning yang dapat memprediksi biaya asuransi kesehatan adalah model dengan algoritma regresi. Dengan algoritma tersebut, model dapat memprediksi biaya asuransi kesehatan berdasarkan karakteristik pemegang polis.
- Di antara beberapa algoritma yang diuji tersebut, algoritma yang dapat memprediksi premi asuransi kesehatan dengan akurasi terbaik adalah algoritma Random Forest. 
