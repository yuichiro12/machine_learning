# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# for scikit-learn <=0.17
#from sklearn.cross_validation import train_test_split
# for scikit-learn 0.18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import precision_recall_fscore_support

# テストデータ分割のための乱数のシード（整数値）
random_seed = 2
# テストデータの割合
test_proportion = 0.2
# Iris データセットをロード  
iris = datasets.load_iris()
# 特徴ベクトルを取得
X = iris.data
# クラスラベルを取得
y = iris.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_proportion, random_state = random_seed)

# Zスコアで正規化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# エントロピーを指標とする決定木のインスタンスを生成し，決定木のモデルに学習データを適合させる
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree.fit(X_train_std, y_train)

y_test_predicted = tree.predict(X_test_std)


# テストデータの正解クラスと決定木による予測クラスを出力
print("Test Data")
print("True Label     ", y_test)
print("Predicted Label", y_test_predicted)

fscore = precision_recall_fscore_support(y_test, y_test_predicted)


print('Class 0 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore[0][0], fscore[1][0], fscore[2][0]))
print('Class 1 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore[0][1], fscore[1][1], fscore[2][1]))
print('Class 2 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore[0][2], fscore[1][2], fscore[2][2]))

# 出力されたtree.dotファイルは，別途Graphviz(gvedit)から開くことで木構造を描画できる
export_graphviz(tree, out_file='tree.dot', feature_names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
print("tree.dot file is generated")
