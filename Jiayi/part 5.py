Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import seaborn as sns
rng = np.random.default_rng(seed=42)
centers = rng.integers(low=3, high=7)
X, y = make_blobs(n_samples=1000, n_features=6, centers=centers, random_state=42)
X = pd.DataFrame(X)
X.head()
           0         1         2         3          4          5
0   6.597330 -5.250127 -6.682249 -7.361722  -4.038499   0.804176
1  -9.754520  6.491701  1.955122  3.445692  -8.906258  10.885443
2  -0.876786  7.584145  4.199834  2.103910  -5.438354  -8.315972
3 -10.205186  7.916090 -0.682091  3.531567 -10.076584  10.031524
4  -1.967735  9.773441  4.063368 -0.617873  -7.425872  -6.488306
X_scaled = StandardScaler().fit_transform(X)
X_scaled[:2,:]
array([[ 1.26318002, -1.30518704, -1.40025528, -1.56153395,  1.10979237,
        -0.02659633],
       [-1.27068889,  0.4467899 ,  0.39416532,  0.73504761, -0.85731322,
         1.47123687]])
kmeans3 = KMeans(n_clusters=3, random_state=42)
kmeans3.fit(X_scaled)
KMeans(n_clusters=3, random_state=42)
print('Clusters: ', kmeans3.labels_)
Clusters:  [1 2 0 2 0 0 0 2 2 1 1 2 1 0 2 2 1 1 1 2 2 0 0 1 0 1 0 1 0 0 2 1 1 2 2 1 0
 1 1 1 2 1 2 2 1 0 2 0 2 0 1 2 1 2 2 1 2 2 0 1 0 2 2 2 1 0 0 1 1 0 2 2 0 2
 2 2 2 1 1 0 1 0 1 1 0 1 2 1 0 2 2 0 1 2 0 0 1 1 0 2 2 0 1 0 0 0 1 0 1 0 1
 2 0 0 2 2 2 0 0 2 2 1 1 1 0 2 2 1 0 1 2 1 2 1 2 1 1 2 2 1 0 2 2 1 0 0 2 2
 0 0 2 2 0 2 1 0 2 2 0 0 0 0 1 1 2 2 2 0 2 0 1 0 1 0 2 2 1 1 0 0 2 1 0 0 0
 1 0 2 1 2 1 2 1 1 1 0 0 2 1 2 0 1 2 2 0 2 2 1 2 1 0 1 0 1 0 1 0 1 2 1 0 2
 0 0 2 2 2 1 0 1 0 0 1 2 2 1 1 0 1 1 0 0 1 2 2 1 1 1 0 1 0 0 1 0 2 1 0 0 1
 1 0 1 0 0 2 1 0 1 1 0 2 1 1 2 2 1 0 1 1 0 2 1 1 1 2 1 2 0 0 0 0 0 2 1 0 2
 0 2 0 1 1 2 0 1 1 1 2 1 1 1 0 1 2 1 2 1 2 2 0 0 0 2 0 2 1 0 1 2 0 2 0 0 0
 0 2 2 2 0 1 1 1 2 2 0 0 0 2 1 1 2 0 2 1 0 2 1 2 2 0 1 0 0 1 0 2 2 2 0 1 1
 1 1 2 1 0 1 0 2 0 2 2 1 0 0 2 2 1 0 1 1 1 1 2 0 0 2 0 0 2 2 0 2 2 1 1 1 2
 1 0 0 1 1 2 2 0 2 1 2 2 0 1 1 2 2 2 0 1 1 2 1 0 0 2 1 1 2 1 0 0 1 0 0 2 0
 0 2 2 1 2 1 0 1 2 2 2 0 0 1 2 0 1 1 2 0 1 1 0 1 0 0 1 2 2 2 0 2 1 0 1 1 1
 1 2 2 1 0 0 2 0 1 2 1 2 1 2 1 2 0 0 1 0 1 0 0 0 0 1 2 2 0 0 2 1 1 2 2 2 0
 2 2 0 1 1 1 1 1 2 2 1 0 2 0 0 1 1 2 0 1 0 0 0 2 1 1 2 2 2 1 1 2 0 2 1 1 0
 1 0 2 2 2 1 2 2 2 2 1 0 1 2 1 1 0 0 2 2 1 2 2 0 0 1 0 1 2 2 1 1 1 0 1 1 1
 1 0 1 1 1 2 2 0 2 2 0 1 2 2 0 2 0 0 2 2 2 0 0 0 2 1 1 0 0 0 2 0 0 1 2 1 1
 1 2 2 1 2 1 1 1 1 0 0 0 0 1 2 1 0 0 2 2 2 0 0 2 2 2 0 2 1 0 1 0 2 0 2 0 2
 0 0 1 0 2 0 0 1 1 2 0 1 0 1 1 1 0 2 0 0 2 1 0 0 1 2 1 0 0 2 1 0 1 2 2 2 1
 0 2 0 1 1 2 2 2 0 2 1 0 0 2 1 2 0 2 1 0 0 1 1 1 0 1 1 2 0 2 1 1 1 1 2 1 2
 2 0 0 2 2 2 1 1 2 1 2 1 2 0 1 1 2 2 2 2 1 0 2 2 2 0 1 0 2 2 0 1 1 2 2 0 1
 2 0 1 1 2 0 2 1 1 1 1 0 1 0 2 2 2 2 2 1 1 2 0 2 0 1 0 0 2 0 1 2 1 0 2 2 2
 0 0 0 0 0 0 1 1 0 0 0 0 0 2 1 1 0 2 1 1 0 0 0 2 1 2 1 1 2 2 2 0 1 2 2 2 2
 2 1 1 0 2 0 1 0 2 2 1 0 1 0 0 0 1 0 2 2 1 2 1 2 1 0 1 2 1 1 0 0 0 0 1 2 2
 0 2 1 1 0 0 2 2 2 0 2 1 0 2 0 2 0 0 0 0 0 0 0 1 0 0 1 0 0 2 2 0 0 2 2 1 2
 0 0 1 0 0 0 2 0 1 2 1 0 2 1 0 2 0 1 0 1 2 0 1 0 2 1 1 1 1 1 2 0 0 2 1 2 2
 1 1 2 0 2 1 1 1 0 1 2 0 1 2 0 0 2 0 0 0 0 2 1 0 0 1 1 2 2 2 2 2 0 2 0 1 1
 0]
print('Inertia: ', kmeans3.inertia_)
Inertia:  306.9478846038064
kmeans3 = KMeans(n_clusters=3, random_state=42)
kmeans3.fit(X_scaled)
KMeans(n_clusters=3, random_state=42)
print('Clusters: ', kmeans3.labels_)
Clusters:  [1 2 0 2 0 0 0 2 2 1 1 2 1 0 2 2 1 1 1 2 2 0 0 1 0 1 0 1 0 0 2 1 1 2 2 1 0
 1 1 1 2 1 2 2 1 0 2 0 2 0 1 2 1 2 2 1 2 2 0 1 0 2 2 2 1 0 0 1 1 0 2 2 0 2
 2 2 2 1 1 0 1 0 1 1 0 1 2 1 0 2 2 0 1 2 0 0 1 1 0 2 2 0 1 0 0 0 1 0 1 0 1
 2 0 0 2 2 2 0 0 2 2 1 1 1 0 2 2 1 0 1 2 1 2 1 2 1 1 2 2 1 0 2 2 1 0 0 2 2
 0 0 2 2 0 2 1 0 2 2 0 0 0 0 1 1 2 2 2 0 2 0 1 0 1 0 2 2 1 1 0 0 2 1 0 0 0
 1 0 2 1 2 1 2 1 1 1 0 0 2 1 2 0 1 2 2 0 2 2 1 2 1 0 1 0 1 0 1 0 1 2 1 0 2
 0 0 2 2 2 1 0 1 0 0 1 2 2 1 1 0 1 1 0 0 1 2 2 1 1 1 0 1 0 0 1 0 2 1 0 0 1
 1 0 1 0 0 2 1 0 1 1 0 2 1 1 2 2 1 0 1 1 0 2 1 1 1 2 1 2 0 0 0 0 0 2 1 0 2
 0 2 0 1 1 2 0 1 1 1 2 1 1 1 0 1 2 1 2 1 2 2 0 0 0 2 0 2 1 0 1 2 0 2 0 0 0
 0 2 2 2 0 1 1 1 2 2 0 0 0 2 1 1 2 0 2 1 0 2 1 2 2 0 1 0 0 1 0 2 2 2 0 1 1
 1 1 2 1 0 1 0 2 0 2 2 1 0 0 2 2 1 0 1 1 1 1 2 0 0 2 0 0 2 2 0 2 2 1 1 1 2
 1 0 0 1 1 2 2 0 2 1 2 2 0 1 1 2 2 2 0 1 1 2 1 0 0 2 1 1 2 1 0 0 1 0 0 2 0
 0 2 2 1 2 1 0 1 2 2 2 0 0 1 2 0 1 1 2 0 1 1 0 1 0 0 1 2 2 2 0 2 1 0 1 1 1
 1 2 2 1 0 0 2 0 1 2 1 2 1 2 1 2 0 0 1 0 1 0 0 0 0 1 2 2 0 0 2 1 1 2 2 2 0
 2 2 0 1 1 1 1 1 2 2 1 0 2 0 0 1 1 2 0 1 0 0 0 2 1 1 2 2 2 1 1 2 0 2 1 1 0
 1 0 2 2 2 1 2 2 2 2 1 0 1 2 1 1 0 0 2 2 1 2 2 0 0 1 0 1 2 2 1 1 1 0 1 1 1
 1 0 1 1 1 2 2 0 2 2 0 1 2 2 0 2 0 0 2 2 2 0 0 0 2 1 1 0 0 0 2 0 0 1 2 1 1
 1 2 2 1 2 1 1 1 1 0 0 0 0 1 2 1 0 0 2 2 2 0 0 2 2 2 0 2 1 0 1 0 2 0 2 0 2
 0 0 1 0 2 0 0 1 1 2 0 1 0 1 1 1 0 2 0 0 2 1 0 0 1 2 1 0 0 2 1 0 1 2 2 2 1
 0 2 0 1 1 2 2 2 0 2 1 0 0 2 1 2 0 2 1 0 0 1 1 1 0 1 1 2 0 2 1 1 1 1 2 1 2
 2 0 0 2 2 2 1 1 2 1 2 1 2 0 1 1 2 2 2 2 1 0 2 2 2 0 1 0 2 2 0 1 1 2 2 0 1
 2 0 1 1 2 0 2 1 1 1 1 0 1 0 2 2 2 2 2 1 1 2 0 2 0 1 0 0 2 0 1 2 1 0 2 2 2
 0 0 0 0 0 0 1 1 0 0 0 0 0 2 1 1 0 2 1 1 0 0 0 2 1 2 1 1 2 2 2 0 1 2 2 2 2
 2 1 1 0 2 0 1 0 2 2 1 0 1 0 0 0 1 0 2 2 1 2 1 2 1 0 1 2 1 1 0 0 0 0 1 2 2
 0 2 1 1 0 0 2 2 2 0 2 1 0 2 0 2 0 0 0 0 0 0 0 1 0 0 1 0 0 2 2 0 0 2 2 1 2
 0 0 1 0 0 0 2 0 1 2 1 0 2 1 0 2 0 1 0 1 2 0 1 0 2 1 1 1 1 1 2 0 0 2 1 2 2
 1 1 2 0 2 1 1 1 0 1 2 0 1 2 0 0 2 0 0 0 0 2 1 0 0 1 1 2 2 2 2 2 0 2 0 1 1
 0]
print('Inertia: ', kmeans3.inertia_)
Inertia:  306.94788460380636
num_clusters = [i for i in range(2, 11)]
def kmeans_inertia(num_clusters, x_vals):
    inertia = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, random_state=42)
        kms.fit(x_vals)
        inertia.append(kms.inertia_)
        return inertia
    inertia = kmeans_inertia(num_clusters, X_scaled)
inertia = kmeans_inertia(num_clusters, X_scaled)
SyntaxError: invalid syntax
     inertia = kmeans_inertia(num_clusters, X_scaled)
     
SyntaxError: unexpected indent
inertia
Traceback (most recent call last):
  File "<pyshell#33>", line 1, in <module>
    inertia
NameError: name 'inertia' is not defined
plot = sns.lineplot(x=num_clusters, y=inertia)
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    plot = sns.lineplot(x=num_clusters, y=inertia)
NameError: name 'inertia' is not defined
plot.set_xlabel("Number of clusters");
Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    plot.set_xlabel("Number of clusters");
NameError: name 'plot' is not defined. Did you mean: 'float'?
plt.figure(figsize=(10, 6))plot = sns.lineplot(x=num_clusters, y=inertia)
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    plot = sns.lineplot(x=num_clusters, y=inertia)
NameError: name 'inertia' is not defined
plot.set_xlabel("Number of clusters");
Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    plot.set_xlabel("Number of clusters");
NameError: name 'plot' is not defined. Did you mean: 'float'?num_clusters = range(1, 11)
SyntaxError: invalid syntax
num_clusters = range(1, 11)
inertia = []
for k in num_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(your_data)
    inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(x=num_clusters, y=inertia)
    plot.set_xlabel("Number of clusters")
    plot.set_ylabel("Inertia")
    plt.show()
plot = sns.lineplot(x=num_clusters, y=inertia)
SyntaxError: invalid syntax
>>>      plot = sns.lineplot(x=num_clusters, y=inertia)
...      
SyntaxError: unexpected indent
>>> kmeans3_sil_score = silhouette_score(X_scaled, kmeans3.labels_)
>>> kmeans3_sil_score
np.float64(0.7804420928014502)
>>> def kmeans_sil(num_clusters, x_vals):
...     sil_score = []
...     for num in num_clusters:
...         kms = KMeans(n_clusters=num, random_state=42)
...         kms.fit(x_vals)
...         sil_score.append(silhouette_score(x_vals, kms.labels_))
... return sil_score
SyntaxError: invalid syntax
>>> sil_score = kmeans_sil(num_clusters, X_scaled)
Traceback (most recent call last):
  File "<pyshell#59>", line 1, in <module>
    sil_score = kmeans_sil(num_clusters, X_scaled)
NameError: name 'kmeans_sil' is not defined
>>> sil_score
Traceback (most recent call last):
  File "<pyshell#60>", line 1, in <module>
    sil_score
NameError: name 'sil_score' is not defined
>>> plot = sns.lineplot(x=num_clusters, y=sil_score)
Traceback (most recent call last):
  File "<pyshell#61>", line 1, in <module>
    plot = sns.lineplot(x=num_clusters, y=sil_score)
NameError: name 'sil_score' is not defined
>>> plot.set_xlabel("# of clusters");
Traceback (most recent call last):
  File "<pyshell#62>", line 1, in <module>
    plot.set_xlabel("# of clusters");
NameError: name 'plot' is not defined. Did you mean: 'float'?
>>> plot.set_ylabel("Silhouette Score");
Traceback (most recent call last):
  File "<pyshell#63>", line 1, in <module>
    plot.set_ylabel("Silhouette Score");
NameError: name 'plot' is not defined. Did you mean: 'float'?
