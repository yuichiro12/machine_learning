#### 知識情報学第10回演習サンプルプログラム ex10.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2016/10/21

#### K-means法によるWineデータのクラスタリング
#### Python機械学習本：11.1 K-means, 11.1.3 Distortion

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter

df_wine = pd.read_csv("wine.data", header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
'OD280/OD315 of diluted wines', 'Proline']

used_feature1 = 'Alcohol'
used_feature2 = 'Malic acid'
X = df_wine[[used_feature1, used_feature2]].values
y = df_wine['Class label'].values
n_class = 3

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#==================================================================
print('K-means')
km = KMeans(n_clusters=4,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

#==================================================================
# 割り当てられたクラスタによりクラスタリング結果を描画

# 描画に使用する色とマークのセット
colors = (["lightgreen", "orange", "lightblue", "m", "b", "g", "c", "y", "w", "k"])
markers = (["s", "o", "v", "^", "D", ">", "<", "d", "p", "H"])

plt.figure(figsize=(8,8))
for idx in range(0, km.cluster_centers_.shape[0]):
    plt.scatter(X[y_km == idx, 0],
                X[y_km == idx, 1],
                s=50,
                c=colors[idx],
                marker=markers[idx],
                label="cluster " + str(idx+1))

plt.scatter(km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    s=250,
    marker='*',
    c='red',
    label='centroids')
plt.legend()
plt.grid()

#==================================================================
# 課題(a) 正解クラスとクラスタ中心を描画
plt.figure(figsize=(8,8))
for idx in range(1, len(set(y))+1):
    plt.scatter(X[y == idx, 0],
                X[y == idx, 1],
                s=50,
                c=colors[idx],
                marker=markers[idx],
                label="cluster " + str(idx))

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.legend()
plt.grid()

#==================================================================
# 内部基準Distortionの算出
print('Distortion: %.2f' % km.inertia_)

# クラスタ数を変えてDistortionをグラフにプロット
plt.figure(figsize=(8,8))
distortions = []
for i in range(1, 11):
    km2 = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km2.fit(X)
    distortions.append(km2.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')

#==================================================================
# 課題(b) 外部基準Purityの算出
# (scikit-learnにPurityは実装されていない)
n_rep_class = []
for val in set(y_km):
    n_rep_class.append(Counter(y[[i for i, v in enumerate(y_km) if v == val]]).most_common(1)[0][1])

purity = sum(n_rep_class) / len(y_km)
print("Purity:", purity)

plt.show()
