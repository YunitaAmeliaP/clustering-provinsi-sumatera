import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, chi2, probplot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Clustering Analysis Sumatera Utara", layout="wide")

# --- Header ---
st.markdown(
    """
    <p style='text-align: center; background-color:#FA5E3C; color: black; 
    padding: 20px; border: 2px solid black; font-weight: bold; font-size: 24px;' >
    Clustering Analysis Sumatera Utara
    </p>
    """, unsafe_allow_html=True
)

# --- 1. Import Data ---
st.markdown("## 1. Import Data")

@st.cache_data
def load_data():
    kpm = pd.read_excel("KPM.xlsx")
    miskin = pd.read_excel("Persentase Penduduk Miskin.xlsx")
    keluhan = pd.read_excel("Persentase Penduduk yang Mempunyai Keluhan Kesehatan Selama Sebulan Terakhir.xlsx")
    sanitasi = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
    minum = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
    sekolah = pd.read_excel("Sekolah.xlsx")
    kerja = pd.read_excel("Tingkat Partisipasi Angkatan Kerja (TPAK).xlsx")
    nganggur = pd.read_excel("Tingkat Pengangguran Terbuka (TPT) Penduduk Umur 15 Tahun Keatas Manurut Kab_Kota.xlsx")
    return kpm, miskin, keluhan, sanitasi, minum, sekolah, kerja, nganggur

kpm, miskin, keluhan, sanitasi, minum, sekolah, kerja, nganggur = load_data()

# --- 2. Preprocessing Data ---
st.markdown("## 2. Preprocessing Data")

kpm = kpm.rename(columns={'JumlahKeluargaPenerimaManfaat(KPM)': 'KPM'})

miskin = miskin.iloc[:, :-2].iloc[:-4, :]
miskin = miskin.rename(columns={'2021': 'Miskin'})
miskin = miskin[miskin['Kabupaten Kota'] != 'Sumatera Utara']

sanitasi = sanitasi.drop(columns=['2019', '2020']).iloc[:-4, :]
sanitasi = sanitasi.rename(columns={'2021': 'Sanitasi'})
sanitasi = sanitasi[sanitasi['Kabupaten Kota'] != 'Sumatera Utara']

minum = minum.drop(columns=['2019', '2020']).iloc[:-4, :]
minum = minum.rename(columns={'2021': 'Minum'})
minum = minum[minum['Kabupaten Kota'] != 'Sumatera Utara']

keluhan = keluhan.drop(columns=['2019', '2020']).iloc[:-4, :]
keluhan = keluhan.rename(columns={'2021': 'Keluhan'})
keluhan = keluhan[keluhan['Kabupaten Kota'] != 'Sumatera Utara']

kerja = kerja.drop(columns=['2022', '2023']).iloc[:-4, :]
kerja = kerja.rename(columns={'2021': 'Kerja'})
kerja = kerja[kerja['Kabupaten Kota'] != 'Sumatera Utara']
kerja["Kabupaten Kota"] = kerja["Kabupaten Kota"].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))

nganggur = nganggur.drop(columns=['2023', '2022']).iloc[:-4, :]
nganggur = nganggur.rename(columns={'2021': 'Nganggur'})
nganggur = nganggur[nganggur['Kabupaten Kota'] != 'Sumatera Utara']

# Merge data berdasarkan "Kabupaten Kota"
data = kpm.merge(sanitasi, on='Kabupaten Kota') \
          .merge(minum, on='Kabupaten Kota') \
          .merge(keluhan, on='Kabupaten Kota') \
          .merge(miskin, on='Kabupaten Kota') \
          .merge(kerja, on='Kabupaten Kota') \
          .merge(sekolah, on='Kabupaten Kota') \
          .merge(nganggur, on='Kabupaten Kota')

st.dataframe(data)

# Cek missing value
if data.isnull().any().any():
    st.warning("Ada missing value di data!")
else:
    st.success("Tidak ada missing value pada data.")

# --- 3. Statistik Deskriptif ---
st.markdown("## 3. Statistik Deskriptif")
st.dataframe(data.describe().T.style.background_gradient(cmap='Blues'))

# --- 4. Distribusi Data ---
st.markdown("## 4. Distribusi Data")

fig1, axs1 = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(data.columns[1:4]):
    sns.histplot(data[col], ax=axs1[i], kde=True)
    axs1[i].set_title(f'Distribusi {col}')
st.pyplot(fig1)

fig2, axs2 = plt.subplots(2, 4, figsize=(20, 10))
axs2 = axs2.flatten()
for i, col in enumerate(data.columns[4:12]):
    sns.histplot(data[col], ax=axs2[i], kde=True)
    axs2[i].set_title(f'Distribusi {col}')
st.pyplot(fig2)

# --- 5. Korelasi antar Fitur ---
st.markdown("## 5. Korelasi antar Fitur")

features = data.columns[1:]
corr = data[features].corr()
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, ax=ax_corr)
st.pyplot(fig_corr)

# --- 6. Uji Normalitas (Shapiro-Wilk) ---
st.markdown("## 6. Uji Normalitas (Shapiro-Wilk)")

alpha = 0.05
for col in features:
    stat, p = shapiro(data[col])
    if p > alpha:
        hasil = "Data Normal"
    else:
        hasil = "Data Tidak Normal"
    st.write(f"{col}: Statistic={stat:.3f}, p-value={p:.3f} => {hasil}")

# --- 7. Uji Poisson untuk Data Diskrit (KPM, Kerja, Nganggur) ---
st.markdown("## 7. Uji Penyebaran Poisson")

diskrit_cols = ['KPM', 'Kerja', 'Nganggur']
for col in diskrit_cols:
    n = len(data[col])
    df = n - 1
    mu = data[col].mean()
    D = ((data[col] - mu)**2).sum() / mu
    chi2_low = chi2.ppf(0.025, df)
    chi2_high = chi2.ppf(0.975, df)
    if (D < chi2_low) or (D > chi2_high):
        hasil = "Ditolak"
    else:
        hasil = "Diterima"
    st.write(f"{col}: D={D:.2f}, Chi2 Low={chi2_low:.2f}, Chi2 High={chi2_high:.2f} => Hipotesis {hasil}")

# --- 8. Q-Q Plot ---
st.markdown("## 8. Q-Q Plot")

fig_qq, axs_qq = plt.subplots(3, 4, figsize=(20, 15))
axs_qq = axs_qq.flatten()
for i, col in enumerate(features):
    probplot(data[col], plot=axs_qq[i])
    axs_qq[i].set_title(f"Q-Q Plot {col}")
st.pyplot(fig_qq)

# --- 9. Scaling Data ---
st.markdown("## 9. Scaling Data")

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
df_scaled = pd.DataFrame(data_scaled, columns=features)
st.dataframe(df_scaled.head())

# --- 10. Clustering K-Means ---
st.markdown("## 10. Clustering K-Means")

# Tentukan jumlah cluster dengan Elbow method
st.markdown("### Menentukan Jumlah Cluster dengan Elbow Method")
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)  # PENTING: pakai data_scaled, bukan X
    inertia.append(kmeans.inertia_)

fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(K, inertia, 'bx-')
ax_elbow.set_xlabel('Jumlah Cluster (k)')
ax_elbow.set_ylabel('Inertia')
ax_elbow.set_title('Elbow Method Untuk Menentukan k')
st.pyplot(fig_elbow)

# User pilih jumlah cluster
n_clusters = st.slider("Pilih Jumlah Cluster (k)", min_value=2, max_value=10, value=3)

kmeans_final = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans_final.fit_predict(data_scaled)

data['Cluster'] = labels

st.markdown(f"### Hasil Cluster dengan k = {n_clusters}")
st.dataframe(data[['Kabupaten Kota', 'Cluster'] + list(features)])

# Visualisasi 2D hasil cluster dengan PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

fig_pca, ax_pca = plt.subplots()
scatter = ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='Set1')
ax_pca.set_xlabel('PC1')
ax_pca.set_ylabel('PC2')
ax_pca.set_title('Visualisasi Klaster Hasil PCA 2D')
legend1 = ax_pca.legend(*scatter.legend_elements(), title="Cluster")
ax_pca.add_artist(legend1)
st.pyplot(fig_pca)
