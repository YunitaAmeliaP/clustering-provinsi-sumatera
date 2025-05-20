import kagglehub

# Download dataset dari Kaggle
dataset_path = kagglehub.dataset_download('wahyuikbalmaulana/dataset-provinsi-sumatera-utara')
print("Dataset berhasil diunduh.")


# 1. Pendahuluan

import streamlit as st

st.image("https://perkim.id/wp-content/uploads/2020/06/sumut-scaled.jpg", caption="Gambar Sumatera Utara")

st.markdown("""
- Berapa banyak kluster yang perlu kita pakai?
- Bagaimana cara mengelompokkan data dengan baik?
""")


# impor barang dari luar negeri

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.stats import pearsonr
from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import shapiro
from scipy.stats import chi2
from scipy.stats import poisson
from termcolor import colored
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

import streamlit as st

def plot_elbow(X, max_k=10):
    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_k+1), sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    plt.show()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# 2. Data
st.markdown(
    '<h3 style="color:white; background-color:#FA5E3C; font-size:150%; letter-spacing:0.5px; padding: 4px;"><b>Import Data</b></h3>',
    unsafe_allow_html=True
)


# Baca data

kpm = pd.read_excel("KPM.xlsx")
Miskin = pd.read_excel("Persentase Penduduk Miskin.xlsx")
Keluhan = pd.read_excel("Persentase Penduduk yang Mempunyai Keluhan Kesehatan Selama Sebulan Terakhir.xlsx")
Sanitasi = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
Minum = pd.read_excel("Persentase Rumah Tangga yang Memiliki Akses Terhadap Sumber Air Minum Layak.xlsx")
Sekolah = pd.read_excel("Sekolah.xlsx")
Kerja = pd.read_excel("Tingkat Partisipasi Angkatan Kerja (TPAK).xlsx")
Nganggur = pd.read_excel("Tingkat Pengangguran Terbuka (TPT) Penduduk Umur 15 Tahun Keatas Manurut Kab_Kota.xlsx")

st.markdown(
    '<h3 style="color:white; background-color:#FA5E3C; font-size:150%; letter-spacing:0.5px; padding: 4px;"><b>Data preprocessing</b></h3>',
    unsafe_allow_html=True
)


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
Kerja = Kerja.drop(Kerja[Kerja['Kabupaten Kota'] == 'Sumatera Utara'].index)

st.markdown("**Kerja bagus :** Setelah processing data, tahap selanjutnya tinggal dimerge lalu diclustering")


#nggabungin data cuy

data = kpm.merge(Sanitasi)
data = data.merge(Minum)
data = data.merge(Keluhan)
data = data.merge(Miskin)
data = data.merge(Kerja)
data = data.merge(Sekolah)
data = data.merge(Nganggur)
data

# Nyari missing value

data[data.isnull().any(axis=1)]

st.markdown("✅ **Kerja bagus :** Ngga ada missing value cuy")

data.describe().T.style.background_gradient(cmap='Blues')

data.dtypes
st.write("Terdapat 3 data diskrit dan 6 data continue")

# 3. EDA

diskrit = data.iloc[:, [1, 7, 8]]

fig=plt.figure(figsize=(15,14))

for i, f in enumerate(diskrit):
    plt.subplot(6, 3, i+1)
    sns.histplot(x=data[f])

    plt.title(f'Feature: {f}')
    plt.xlabel('')

fig.suptitle('Diskrit feature distributions',  size=20)
fig.tight_layout()
plt.show()


continues = data.iloc[:, [2, 3, 4, 5, 6, 9]]

fig=plt.figure(figsize=(15,14))

for i, f in enumerate(continues):
    plt.subplot(6, 6, i+1)
    sns.histplot(x=data[f])

    plt.title(f'Feature: {f}')
    plt.xlabel('')

fig.suptitle('Continuous feature distributions',  size=20)
fig.tight_layout()
plt.show()


data_test = data.iloc[:, 1:]

def correlation_matrix(dataframe, x, y):
    corr = dataframe.corr()
    f, ax = plt.subplots(figsize=(x, y))
    mask = np.triu(np.ones_like(corr))
    sns.heatmap(corr, annot=True, mask = mask, cmap="Reds")
    return ax

correlation_matrix(data_test, 7, 7)

# 4. Hypothesis testing

## **Uji Shapiro-Wilk**
st.write("(Diambil dari [Francisco Javier Gallego & Torch me](https://www.kaggle.com/code/javigallego/outliers-eda-clustering-tutorial))")

# normality test
stat, p = shapiro(data_test)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret results
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

# Univariate normality test
for col in data_test.columns:
    stat, p = shapiro(data_test[col])
    alpha = 0.05    # significance level
    if p > alpha:
        result = colored('Accepted', 'green')
    else:
        result = colored('Rejected','red')
    print('Feature: {}\t Hypothesis: {}'.format(col, result))

## **Uji Penyebaran Poisson**

# Univariate poisson test
for col in diskrit.columns:
    # Parameters
    alpha = 0.05                  # significance level
    n = len(data[col])            # sample size
    df = n-1                      # degrees of freedom

    # Statistics
    mu = data[col].mean()               # sample mean
    D = ((data[col]-mu)**2).sum()/mu    # test statistic

    # Two-tailed test
    q_lower = alpha/2
    q_upper = (1-alpha)/2

    # percentile point function = inverse of cdf
    chi2_crit_lower = chi2.ppf(q_lower, df)
    chi2_crit_upper = chi2.ppf(q_upper, df)

    if (D<chi2_crit_lower) or (D>chi2_crit_upper):
        result = colored('Rejected', 'red')
    else:
        result = colored('Accepted', 'green')
    print('Feature: {}\t Hypothesis: {}'.format(col, result))
    # print('D:',int(D),', chi2_crit_lower:',int(chi2_crit_lower),', chi2_crit_upper:',int(chi2_crit_upper),'\n')

# Normal Q-Q plots
figure = plt.figure(figsize = (16,12))
for i in range(len(data_test.columns)):

    # Q-Q plot
    ax = plt.subplot(6,5, i+1)
    stats.probplot(data_test.iloc[:,i], dist='norm', plot=plt)

    # Aesthetics
    ax.get_lines()[0].set_markersize(6.0)
    ax.get_lines()[1].set_linewidth(3.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title(data_test.columns[i])

figure.tight_layout(h_pad=1.0, w_pad=0.5)
plt.suptitle('Normal Q-Q Charts', y=1.02, fontsize=20)
plt.show()

# 5. Outlier Detection
### Uji Grubbs


import scipy.stats as stats

def grubbs_test(x, feature):
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    numerator = max(abs(x-mean_x))
    g_calculated = numerator/sd_x
    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    if g_critical > g_calculated:
        result = colored('Accepted', 'green')
        outlier_presence = False
    else:
        result = colored('Rejected','red')
        outlier_presence = True

    print('Feature: {}\t Hypothesis: {}'.format(feature, result))
    return outlier_presence

outliers_features = []
for col in data_test.columns:
    if grubbs_test(data_test[col], col):
        outliers_features.append(col)


# Thus, the following features have outliers on their data
np.transpose(outliers_features)

### Metode Z-Score

def Zscore_outlier(df):
    out=[]
    m = np.mean(df)
    sd = np.std(df)
    row = 0
    for i in df:
        z = (i-m)/sd
        if np.abs(z) > 3:
            out.append(row)
        row += 1
    return out

for col in outliers_features:
    percentage = round(100 * len(Zscore_outlier(data_test[col])) / data_test.shape[0], 2)
    print('Feature: {} \tPercentage of outliers: {}%'.format(col, percentage))

## Isolation Forest

from sklearn.ensemble import IsolationForest

fig, axs = plt.subplots(1, 3, figsize=(22, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()

for i, column in enumerate(data_test[outliers_features].columns):
    isolation_forest = IsolationForest(contamination='auto')
    isolation_forest.fit(data_test[column].values.reshape(-1,1))

    xx = np.linspace(data_test[column].min(), data_test[column].max(), len(data_test)).reshape(-1,1)
    anomaly_score = isolation_forest.decision_function(xx)
    outlier = isolation_forest.predict(xx)

    axs[i].plot(xx, anomaly_score, label='anomaly score')
    axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                     where=outlier==-1, color='r',
                     alpha=.4, label='outlier region')
    axs[i].legend()
    axs[i].set_title(column)

# 6. Cluster Modelling

Provinsi = data.iloc[:, 0]
Atribut = data.iloc[:, 1:]

Atribut_std = pd.DataFrame(StandardScaler().fit_transform(Atribut))
#Atribut_std = pd.DataFrame(RobustScaler().fit_transform(data))
#Atribut_std = pd.DataFrame(PowerTransformer().fit_transform(data))

Atribut_std.columns = data_test.columns

pca = PCA(n_components = 2,random_state = 42)
pca_data= pd.DataFrame(pca.fit_transform(Atribut_std), columns=(["PCA1","PCA2"]))
pca_provinsi = pd.concat([Provinsi, pca_data], axis=1)
pca_provinsi

knn = NearestNeighbors(n_neighbors = 7)
model = knn.fit(Atribut_std)
distances, indices = knn.kneighbors(Atribut_std)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.grid()
plt.plot(distances);
plt.xlabel('Points Sorted by Distance')
plt.ylabel('7-NN Distance')
plt.title('K-Distance Graph');

db = DBSCAN(eps = 6, min_samples = 6).fit(Atribut_std)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Number of Clusters : ', n_clusters_)
print('Number of Outliers : ', n_noise_)

#data['Class'] = labels; Atribut_std['Class'] = labels

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering().fit(Atribut_std)

from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(Atribut_std)
fignum = 1

plt.figure(fignum, figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

linkage_data = linkage(Atribut_std, method = 'ward', metric = 'euclidean')
dendrogram(linkage_data)
plt.tight_layout()
plt.show()

linkage_data = linkage(pca_data, method = 'ward', metric = 'euclidean')
dendrogram(linkage_data)
plt.tight_layout()
plt.show()

model_gmm = GaussianMixture(n_components=4, random_state=0).fit(Atribut_std)
gmm_labels = pd.DataFrame(model_gmm.predict(Atribut_std))

haha = pd.concat((data, gmm_labels), axis=1)
haha = haha.rename(columns={0:'gmm_cluster'})
haha['gmm_cluster'] = haha['gmm_cluster'] + 1

haha.head()

### Metode Elbow

# Instantiate the KMeans model and visualizer
model = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=42)
visualizer = KElbowVisualizer(model, k=(1,11))

# Fit the data and visualize
visualizer.fit(Atribut_std)
visualizer.show()

# Instantiate the KMeans model and visualizer
model = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=42)
visualizer = KElbowVisualizer(model, k=(1,11))

# Fit the data and visualize
visualizer.fit(pca_data)
visualizer.show()

### Silhouette score

import sklearn.metrics as metrics
for i in range(4,13):
    labels=KMeans(n_clusters=i,init="k-means++",random_state=200).fit(Atribut_std).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(Atribut_std,labels,metric="euclidean",sample_size=1000,random_state=200)))

import sklearn.metrics as metrics
for i in range(4,13):
    labels=KMeans(n_clusters=i,init="k-means++",random_state=200).fit(pca_data).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(pca_data,labels,metric="euclidean",sample_size=1000,random_state=200)))

from sklearn.cluster import KMeans

# Create the KMeans model and fit it to the standardized data
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(Atribut_std)
labels = pd.DataFrame(kmeans.labels_)

print(pd.Series(kmeans.labels_).value_counts())
#print(labels)


'''
atau
labels = kmeans.fit_predict(Atribut_std)
labels
'''

hehe = pd.concat((data,labels),axis=1)
hehe = hehe.rename({0:'kmeans_cluster'},axis=1)
hehe['kmeans_cluster'] = hehe['kmeans_cluster'] + 1

hehe.head()

# PCA
km4 = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(pca_data)

pca_provinsi['Labels'] = km4.labels_
pca_provinsi['Labels'] = pca_provinsi['Labels'] + 1

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_provinsi['PCA1'], y=pca_provinsi['PCA2'], hue=pca_provinsi['Labels'],
                palette=sns.color_palette('hls', 4))

plt.title('KMeans with 4 Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# 7. Deeper Analysis using K-Means

# Feature Combination : presentase kemiskinan + presentase keluhan kesehatan + presentase pengangguran
m1 = data_test[['Miskin', 'Keluhan', 'Nganggur']]
m1_std = pd.DataFrame(StandardScaler().fit_transform(m1))
m1_std.columns = m1.columns

# Instantiate the KMeans model and visualizer
model = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=42)
visualizer = KElbowVisualizer(model, k=(1,11))

# Fit the data and visualize
visualizer.fit(m1_std)
visualizer.show()

model = KMeans(n_clusters = 3, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0).fit(m1_std)
labels = pd.DataFrame(model.labels_)

# Untuk keperluan visualisasi

hehe = pd.concat((m1_std,labels),axis=1)
hehe = hehe.rename({0:'kmeans_cluster'},axis=1)
hehe['kmeans_cluster'] = hehe['kmeans_cluster'] + 1

hehe.head()

# Untuk keperluan data

heha = pd.concat((m1,labels),axis=1)
heha = heha.rename({0:'kmeans_cluster'},axis=1)
heha['kmeans_cluster'] = heha['kmeans_cluster'] + 1

heha

## 0 : No Help Needed
## 1 : Help Needed
## 2 : Might Need Help

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

colors = {1: 'r', 2: 'g', 3: 'b'}
for cluster, color in colors.items():
    cluster_df = hehe[hehe['kmeans_cluster'] == cluster]
    ax.scatter(cluster_df['Miskin'], cluster_df['Keluhan'], cluster_df['Nganggur'], c=color, label=f'Cluster {cluster}')

ax.set_xlabel('Miskin')
ax.set_ylabel('Keluhan')
ax.set_zlabel('Nganggur')
ax.set_title('Clustering Visualization in 3D')
ax.legend()

plt.show()


fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))

plt.subplot(1,3,1)
sns.boxplot(x = 'kmeans_cluster', y = 'Miskin', data  = heha, color = '#FF781F');
plt.title('Persentase kemiskinan')

plt.subplot(1,3,2)
sns.boxplot(x = 'kmeans_cluster', y = 'Nganggur', data  = heha, color = '#FF781F');
plt.title('Persentase Pengangguran')

plt.subplot(1,3,3)
sns.boxplot(x = 'kmeans_cluster', y = 'Keluhan', data  = heha, color = '#FF781F');
plt.title('Persentase keluhan kesehatan')

plt.show()

# Feature Combination : presentase kemiskinan + presentase keluhan kesehatan + presentase pengangguran
m2 = data_test[['Sanitasi', 'Minum']]
m2_std = pd.DataFrame(StandardScaler().fit_transform(m2))
m2_std.columns = m2.columns

# Instantiate the KMeans model and visualizer
model = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=42)
visualizer = KElbowVisualizer(model, k=(1,11))

# Fit the data and visualize
visualizer.fit(m2_std)
visualizer.show()

# Instantiate the KMeans model and visualizer
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42).fit(m2_std)
m2_std['Labels'] = kmeans.labels_
m2_std['Labels'] = m2_std['Labels'] + 1

plt.figure(figsize=(8, 6))
sns.scatterplot(x=m2_std['Sanitasi'], y=m2_std['Minum'], hue=m2_std['Labels'],
                palette=sns.color_palette('hls', 3))

plt.title('KMeans with 3 Clusters')
plt.xlabel('Sanitasi')
plt.ylabel('Minum')
plt.show()

# Hasil akhir

# Kesimpulan
