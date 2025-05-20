# -----------------------------------------------
# STREAMLIT APP: KLUSTERING DATA PROVINSI SUMATERA UTARA
# -----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------
# JUDUL DAN GAMBAR
# -----------------------------------------------
st.title("Klasterisasi Data Sosial Ekonomi Sumatera Utara")
st.image("https://perkim.id/wp-content/uploads/2020/06/sumut-scaled.jpg", caption="Gambar Sumatera Utara", use_column_width=True)

st.markdown("""
**Tujuan Analisis:**
- Menentukan jumlah kluster optimal
- Mengelompokkan kabupaten/kota berdasarkan indikator sosial ekonomi
""")

# -----------------------------------------------
# IMPORT DATA
# -----------------------------------------------
st.markdown(
    '<h3 style="color:white; background-color:#FA5E3C; font-size:150%; letter-spacing:0.5px; padding: 4px;"><b>Import Data</b></h3>',
    unsafe_allow_html=True
)

kpm = pd.read_excel("KPM.xlsx")
miskin = pd.read_excel("Persentase Penduduk Miskin.xlsx")
keluhan = pd.read_excel("Persentase Penduduk yang Mempunyai Keluhan Kesehatan Selama Sebulan Terakhir.xlsx")
sanitasi = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
minum = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
sekolah = pd.read_excel("Sekolah.xlsx")
kerja = pd.read_excel("Tingkat Partisipasi Angkatan Kerja (TPAK).xlsx")
nganggur = pd.read_excel("Tingkat Pengangguran Terbuka (TPT) Penduduk Umur 15 Tahun Keatas Manurut Kab_Kota.xlsx")

# -----------------------------------------------
# PREPROCESSING
# -----------------------------------------------
st.markdown(
    '<h3 style="color:white; background-color:#FA5E3C; font-size:150%; letter-spacing:0.5px; padding: 4px;"><b>Data Preprocessing</b></h3>',
    unsafe_allow_html=True
)

kpm = kpm.rename(columns={'JumlahKeluargaPenerimaManfaat(KPM)': 'KPM'})

# Bersihkan data dan ambil kolom tahun 2021
def bersihkan_data(df, tahun, nama_baru):
    df = df.drop(columns=[col for col in df.columns if col != 'Kabupaten Kota' and col != tahun])
    df = df.rename(columns={tahun: nama_baru})
    df = df[~df['Kabupaten Kota'].str.contains('Sumatera Utara', na=False)]
    df = df.iloc[:-4, :]  # buang baris bawah (total/provinsi)
    return df

miskin = bersihkan_data(miskin, '2021', 'Miskin')
keluhan = bersihkan_data(keluhan, '2021', 'Keluhan')
sanitasi = bersihkan_data(sanitasi, '2021', 'Sanitasi')
minum = bersihkan_data(minum, '2021', 'Minum')
kerja = bersihkan_data(kerja, '2021', 'Kerja')
nganggur = bersihkan_data(nganggur, '2021', 'Nganggur')

# Format Nama Kabupaten
kerja["Kabupaten Kota"] = kerja["Kabupaten Kota"].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
nganggur["Kabupaten Kota"] = nganggur["Kabupaten Kota"].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
kpm["Kabupaten Kota"] = kpm["Kabupaten Kota"].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))

# Gabungkan semua data
data = kpm.merge(sanitasi, on='Kabupaten Kota') \
          .merge(minum, on='Kabupaten Kota') \
          .merge(keluhan, on='Kabupaten Kota') \
          .merge(miskin, on='Kabupaten Kota') \
          .merge(kerja, on='Kabupaten Kota') \
          .merge(sekolah, on='Kabupaten Kota') \
          .merge(nganggur, on='Kabupaten Kota')

st.write("✅ Data berhasil digabung.")
st.dataframe(data)

# Cek missing value
if data.isnull().sum().sum() == 0:
    st.markdown("✅ **Tidak ada missing value.**")
else:
    st.markdown("⚠️ **Ada data yang hilang.**")

# -----------------------------------------------
# EDA
# -----------------------------------------------
st.subheader("Exploratory Data Analysis")

diskrit = data.iloc[:, [1, 7, 8]]
continues = data.iloc[:, [2, 3, 4, 5, 6, 9]]

# Histogram Diskrit
st.markdown("**Distribusi Fitur Diskrit**")
fig1 = plt.figure(figsize=(15, 5))
for i, f in enumerate(diskrit):
    plt.subplot(1, 3, i + 1)
    sns.histplot(x=diskrit[f])
    plt.title(f)
st.pyplot(fig1)

# Histogram Continues
st.markdown("**Distribusi Fitur Kontinu**")
fig2 = plt.figure(figsize=(15, 10))
for i, f in enumerate(continues):
    plt.subplot(2, 3, i + 1)
    sns.histplot(x=continues[f])
    plt.title(f)
st.pyplot(fig2)

# Correlation Matrix
st.markdown("**Matriks Korelasi**")
def correlation_matrix(df, x=7, y=7):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr))
    f, ax = plt.subplots(figsize=(x, y))
    sns.heatmap(corr, annot=True, mask=mask, cmap="Reds")
    st.pyplot(f)

correlation_matrix(data.iloc[:, 1:])

# -----------------------------------------------
# SHAPIRO-WILK NORMALITY TEST
# -----------------------------------------------
st.subheader("Uji Normalitas Shapiro-Wilk")

hasil_normalitas = []
for col in data.columns[1:]:
    stat, p = shapiro(data[col])
    hasil = "Diterima (Normal)" if p > 0.05 else "Ditolak (Tidak Normal)"
    hasil_normalitas.append((col, p, hasil))

shapiro_df = pd.DataFrame(hasil_normalitas, columns=["Fitur", "p-value", "Kesimpulan"])
st.dataframe(shapiro_df)

# -----------------------------------------------
# ELBOW METHOD
# -----------------------------------------------
st.subheader("Menentukan Jumlah Kluster dengan Elbow Method")

scaled_data = StandardScaler().fit_transform(data.iloc[:, 1:])

def plot_elbow(X, max_k=10):
    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, max_k+1), sse, marker='o')
    ax.set_xlabel('Jumlah Kluster (k)')
    ax.set_ylabel('SSE')
    ax.set_title('Metode Elbow')
    st.pyplot(fig)

plot_elbow(scaled_data)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Salin data untuk klasterisasi
data_cluster = all_data_cleaned.drop(columns=['Kabupaten/Kota'])

# Standarisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cluster)

# Inisialisasi dan fit KMeans
k = 4  # Jumlah kluster dari hasil elbow
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(data_scaled)

# Tambahkan hasil klaster ke dataframe
all_data_cleaned['Klaster'] = kmeans.labels_

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Reduksi dimensi agar bisa divisualisasikan (PCA ke 2 dimensi)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Buat DataFrame untuk visualisasi
df_vis = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
df_vis['Klaster'] = kmeans.labels_
df_vis['Kabupaten/Kota'] = all_data_cleaned['Kabupaten/Kota']

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='Klaster', palette='Set2', s=100)
for i in range(df_vis.shape[0]):
    plt.text(df_vis['PC1'][i]+0.1, df_vis['PC2'][i], df_vis['Kabupaten/Kota'][i], fontsize=9)
plt.title('Visualisasi Klasterisasi Kabupaten/Kota')
plt.legend(title='Klaster')
plt.grid(True)
plt.show()

# Simpan ke file Excel
all_data_cleaned.to_excel("Hasil_Klasterisasi_KabupatenKota.xlsx", index=False)
print("Hasil klasterisasi berhasil disimpan ke 'Hasil_Klasterisasi_KabupatenKota.xlsx'")




