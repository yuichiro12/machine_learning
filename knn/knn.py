####-*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support

# K近傍法の近傍数パラメータ k
neighbors = 5
# テストデータ分割のための乱数のシード（整数値）
random_seed = 1
#　テストデータの割合
test_proportion = 0.3
# Iris データセットをロード 
iris = datasets.load_iris()
# 使用する特徴の次元を(Irisの場合は0,1,2,3から)2つ指定．d1とd2は異なる次元を指定すること
d1 = 0
d2 = 1
# d1,d2列目の特徴量を使用 
X = iris.data[:, [d1, d2]]
# クラスラベルを取得
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=random_seed)


# 各特徴毎に平均0，標準偏差1に標準化（zスコアとも呼ばれる）
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# クラスKNeighborsClassifierを使用してk近傍法のインスタンスknnを生成
knn = KNeighborsClassifier(n_neighbors=neighbors)

knn.fit(X_train_std, y_train)

# 結果をプロット
x1_min, x1_max = X_train_std[:, 0].min() - 0.5, X_train_std[:, 0].max() + 0.5
x2_min, x2_max = X_train_std[:, 1].min() - 0.5, X_train_std[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))
                       
Z = knn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)

markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])

plt.figure(figsize=(10,10))
plt.subplot(211)

plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y_train)):
    plt.scatter(x=X_train_std[y_train == cl, 0], y=X_train_std[y_train == cl, 1],
                alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)


plt.xlabel('sepal length [standardized]')
plt.ylabel('sepal width [standardized]')
plt.title('train_data')

plt.subplot(212)

plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=X_test_std[y_test == cl, 0], y=X_test_std[y_test == cl, 1],
                alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)


plt.xlabel('sepal length [standardized]')
plt.ylabel('sepal width [standardized]')
plt.title('test_data')
plt.show()

