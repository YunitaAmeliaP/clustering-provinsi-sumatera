# Import libraries
import kagglehub

# Download dataset dari Kaggle
dataset_path = kagglehub.dataset_download('wahyuikbalmaulana/dataset-provinsi-sumatera-utara')
print("Dataset berhasil diunduh.")


# Impor library lain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, chi2
from termcolor import colored
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Pilih kolom fitur yang mau dipakai clustering, misalnya kolom 'feature1' dan 'feature2'
X = df[['feature1', 'feature2']].values  # numpy array untuk fitting

# Buat list untuk nyimpen nilai inertia (jumlah jarak kuadrat dalam cluster)
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan Jumlah Cluster')
plt.show()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ===== 2. Data =====

# Baca data
kpm = pd.read_excel("KPM.xlsx")
Miskin = pd.read_excel("Persentase Penduduk Miskin.xlsx")
Keluhan = pd.read_excel("Persentase Penduduk yang Mempunyai Keluhan Kesehatan Selama Sebulan Terakhir.xlsx")
Sanitasi = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
Minum = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
Sekolah = pd.read_excel("Sekolah.xlsx")
Kerja = pd.read_excel("Tingkat Partisipasi Angkatan Kerja (TPAK).xlsx")
Nganggur = pd.read_excel("Tingkat Pengangguran Terbuka (TPT) Penduduk Umur 15 Tahun Keatas Manurut Kab_Kota.xlsx")


# Data preprocessing tiap dataset
kpm = kpm.rename(columns={'JumlahKeluargaPenerimaManfaat(KPM)':'KPM'})

Miskin = Miskin.iloc[:, :-2].iloc[:-4,:]
Miskin = Miskin.rename(columns={'2021':'Miskin'})
Miskin = Miskin.drop(Miskin[Miskin['Kabupaten Kota'] == 'Sumatera Utara'].index)

Sanitasi = Sanitasi.drop(columns=['2019', '2020']).iloc[:-4,:]
Sanitasi = Sanitasi.rename(columns={'2021':'Sanitasi'})
Sanitasi = Sanitasi.drop(Sanitasi[Sanitasi['Kabupaten Kota'] == 'Sumatera Utara'].index)

Minum = Minum.drop(columns=['2019', '2020']).iloc[:-4,:]
Minum = Minum.rename(columns={'2021':'Minum'})
Minum = Minum.drop(Minum[Minum['Kabupaten Kota'] == 'Sumatera Utara'].index)

Keluhan = Keluhan.drop(columns=['2019', '2020']).iloc[:-4,:]
Keluhan = Keluhan.rename(columns={'2021':'Keluhan'})
Keluhan = Keluhan.drop(Keluhan[Keluhan['Kabupaten Kota'] == 'Sumatera Utara'].index)

Kerja = Kerja.drop(columns=['2022', '2023']).iloc[:-4,:]
Kerja = Kerja.rename(columns={'2021':'Kerja'})
Kerja = Kerja.drop(Kerja[Kerja['Kabupaten Kota'] == 'Sumatera Utara'].index)
Kerja["Kabupaten Kota"] = Kerja["Kabupaten Kota"].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))

Nganggur = Nganggur.drop(columns=['2023', '2022']).iloc[:-4,:]
Nganggur = Nganggur.rename(columns={'2021':'Nganggur'})
Nganggur = Nganggur.drop(Nganggur[Nganggur['Kabupaten Kota'] == 'Sumatera Utara'].index)


# Merge semua data
data = kpm.merge(Sanitasi)
data = data.merge(Minum)
data = data.merge(Keluhan)
data = data.merge(Miskin)
data = data.merge(Kerja)
data = data.merge(Sekolah)
data = data.merge(Nganggur)
print("Data gabungan:")
print(data.head())

# Cek missing value
print("Missing values di tiap baris:")
print(data[data.isnull().any(axis=1)])

# Deskriptif statistik
print(data.describe().T)

# Cek tipe data
print(data.dtypes)


# ===== 3. EDA =====

# Data diskrit dan kontinyu
diskrit = data.iloc[:, [1, 7, 8]]
continues = data.iloc[:, [2, 3, 4, 5, 6, 9]]

# Visualisasi distribusi diskrit
fig=plt.figure(figsize=(15,14))
for i, f in enumerate(diskrit):
    plt.subplot(6, 3, i+1)
    sns.histplot(x=data[f])
    plt.title(f'Feature: {f}')
    plt.xlabel('')
fig.suptitle('Diskrit feature distributions',  size=20)
fig.tight_layout()
plt.show()

# Visualisasi distribusi kontinu
fig=plt.figure(figsize=(15,14))
for i, f in enumerate(continues):
    plt.subplot(6, 6, i+1)
    sns.histplot(x=data[f])
    plt.title(f'Feature: {f}')
    plt.xlabel('')
fig.suptitle('Continuous feature distributions',  size=20)
fig.tight_layout()
plt.show()

# Korelasi matrix
data_test = data.iloc[:, 1:]
def correlation_matrix(dataframe, x, y):
    corr = dataframe.corr()
    f, ax = plt.subplots(figsize=(x, y))
    mask = np.triu(np.ones_like(corr))
    sns.heatmap(corr, annot=True, mask = mask, cmap="Reds")
    return ax
correlation_matrix(data_test, 7, 7)
plt.show()


# ===== 4. Uji Hipotesis =====

# Uji Shapiro-Wilk (normalitas) untuk seluruh data sekaligus
stat, p = shapiro(data_test)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

# Uji normalitas per fitur
for col in data_test.columns:
    stat, p = shapiro(data_test[col])
    if p > alpha:
        result = colored('Accepted', 'green')
    else:
        result = colored('Rejected','red')
    print(f'Feature: {col}\t Hypothesis: {result}')

# Uji Poisson untuk data diskrit
for col in diskrit.columns:
    n = len(data[col])
    df = n-1
    mu = data[col].mean()
    D = ((data[col]-mu)**2).sum()/mu
    q_lower = alpha/2
    q_upper = 1 - alpha/2
    chi2_crit_lower = chi2.ppf(q_lower, df)
    chi2_crit_upper = chi2.ppf(q_upper, df)
    if (D < chi2_crit_lower) or (D > chi2_crit_upper):
        result = colored('Rejected', 'red')
    else:
        result = colored('Accepted', 'green')
    print(f'Feature: {col}\t Hypothesis: {result}')

# Normal Q-Q plots untuk tiap fitur kontinyu
figure = plt.figure(figsize = (16,12))
for i in range(len(data_test.columns)):
    ax = plt.subplot(6,5, i+1)
    stats.probplot(data_test.iloc[:,i], dist='norm', plot=plt)
    ax.get_lines()[0].set_markersize(6.0)
    plt.title(data_test.columns[i])
plt.tight_layout()
plt.show()


# ===== 5. PREPROCESSING UNTUK CLUSTERING =====

from sklearn.preprocessing import StandardScaler

# Pilih fitur numerik yang akan di-cluster (kecuali nama daerah dan variabel target jika ada)
features = data.iloc[:, 1:]  # ambil semua kolom kecuali nama daerah

# Standarisasi data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("Data berhasil diskalakan.")


# ===== 6. MENENTUKAN JUMLAH CLUSTER OPTIMAL =====

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(features_scaled)
visualizer.show()

print(f"Jumlah cluster optimal menurut Elbow Method adalah: {visualizer.elbow_value_}")


# ===== 7. CLUSTERING MENGGUNAKAN K-MEANS =====

# Misal jumlah cluster optimal = 4 (ganti sesuai hasil elbow)
k_optimal = visualizer.elbow_value_

kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Tambahkan hasil cluster ke data asli
data['Cluster'] = clusters

print(data[['Kabupaten Kota', 'Cluster']].head())


# ===== 8. VISUALISASI HASIL CLUSTERING =====

import matplotlib.pyplot as plt
import seaborn as sns

# Plot scatter 2D menggunakan 2 fitur utama (misal 2 fitur pertama)
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=features.iloc[:,0], 
    y=features.iloc[:,1], 
    hue=data['Cluster'], 
    palette='Set2',
    s=100
)
plt.title('Visualisasi Cluster berdasarkan 2 fitur pertama')
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.legend(title='Cluster')
plt.show()

# Atau bisa juga pakai PCA untuk visualisasi 2D dari data multi-dimensi
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
components = pca.fit_transform(features_scaled)

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=components[:,0], 
    y=components[:,1], 
    hue=data['Cluster'], 
    palette='Set2',
    s=100
)
plt.title('Visualisasi Cluster dengan PCA 2 Komponen')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()


# ===== 5. PREPROCESSING UNTUK CLUSTERING =====

from sklearn.preprocessing import StandardScaler

# Pilih fitur numerik yang akan di-cluster (kecuali nama daerah dan variabel target jika ada)
features = data.iloc[:, 1:]  # ambil semua kolom kecuali nama daerah

# Standarisasi data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("Data berhasil diskalakan.")


# ===== 6. MENENTUKAN JUMLAH CLUSTER OPTIMAL =====

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(features_scaled)
visualizer.show()

print(f"Jumlah cluster optimal menurut Elbow Method adalah: {visualizer.elbow_value_}")


# ===== 7. CLUSTERING MENGGUNAKAN K-MEANS =====

# Misal jumlah cluster optimal = 4 (ganti sesuai hasil elbow)
k_optimal = visualizer.elbow_value_

kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Tambahkan hasil cluster ke data asli
data['Cluster'] = clusters

print(data[['Kabupaten Kota', 'Cluster']].head())


# ===== 8. VISUALISASI HASIL CLUSTERING =====

import matplotlib.pyplot as plt
import seaborn as sns

# Plot scatter 2D menggunakan 2 fitur utama (misal 2 fitur pertama)
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=features.iloc[:,0], 
    y=features.iloc[:,1], 
    hue=data['Cluster'], 
    palette='Set2',
    s=100
)
plt.title('Visualisasi Cluster berdasarkan 2 fitur pertama')
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.legend(title='Cluster')
plt.show()

# Atau bisa juga pakai PCA untuk visualisasi 2D dari data multi-dimensi
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
components = pca.fit_transform(features_scaled)

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=components[:,0], 
    y=components[:,1], 
    hue=data['Cluster'], 
    palette='Set2',
    s=100
)
plt.title('Visualisasi Cluster dengan PCA 2 Komponen')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()

# ===== 9. ANALISIS KARAKTERISTIK TIAP CLUSTER =====

# Melihat rata-rata nilai setiap fitur di tiap cluster
cluster_summary = data.groupby('Cluster').mean().reset_index()

print("Rata-rata nilai fitur per cluster:")
print(cluster_summary)

# Kalau mau, tampilkan juga jumlah data tiap cluster
print("\nJumlah data per cluster:")
print(data['Cluster'].value_counts())

# Visualisasi perbandingan rata-rata fitur tiap cluster (bar plot)
plt.figure(figsize=(12,8))
sns.barplot(data=cluster_summary.melt(id_vars='Cluster'), 
            x='variable', y='value', hue='Cluster')
plt.title('Rata-rata Nilai Fitur per Cluster')
plt.xticks(rotation=45)
plt.show()
