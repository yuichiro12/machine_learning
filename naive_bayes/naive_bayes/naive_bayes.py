import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer,label_binarize,LabelEncoder,OneHotEncoder
from scipy.io import arff
from sklearn import cross_validation
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

# arffデータの読み込み
f = open("weather.nominal.arff", "r", encoding="utf-8")
data, meta = arff.loadarff(f)

# ラベルエンコーダの設定
le = [LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(),LabelEncoder()]
for idx,attr in enumerate(meta):
    le[idx].fit(meta._attributes[attr][1])

# 特徴ベクトルとクラスラベルの取得とエンコード
# カテゴリ変数を数値データに符号化する必要がある
## （参考）あるカテゴリ変数の主なエンコード方法
## (1) K個の名称を整数値(0,..,k-1)に置き換える
## (2) K個の名称をKビットで表現し，ひとつだけ1，残り全て0とする（1-of-K表現）
##     例：color変数=buleの場合； [blue, green, red] = [1, 0, 0]
## 注意：(1)の符号化法はクラスラベルの符号化や，カテゴリの変数値が順序を持つ場合に用いる方法
##      (2)の符号化法はカテゴリの変数値が順序を持たない場合に用いる方法

feature = []
class_label = []
for x in data:
    w = list(x)
    tmp = le[-1].transform([w[-1].decode("utf-8")])
    class_label.append(tmp[0])
    w.pop(-1)
    for idx in range(0, len(w)):
        tmp2 = le[idx].transform([w[idx].decode("utf-8")])
        w[idx] = tmp2[0]
    feature.append(w)

feature_array = np.array(feature)
class_array = np.array(class_label)

enc = OneHotEncoder()
feature_encoded = enc.fit_transform(feature_array).toarray()


print("Leave-one-out Cross-validation")
y_train_post_list = []
y_train_list = []
y_test_post_list = []
y_test_list = []

loo = cross_validation.LeaveOneOut(len(class_label))
for train_index, test_index in loo:
    X_train, X_test = feature_array[train_index], feature_array[test_index]
    y_train, y_test = class_array[train_index], class_array[test_index]
    
    clf = MultinomialNB(alpha=0.001, class_prior=[0.75, 0.25])
    #clf = MultinomialNB(class_prior=[.6, .4])
    clf.fit(X_train, y_train)

    posterior_trn = clf.predict_proba(X_train)
    posterior_tst = clf.predict_proba(X_test)
    
    print("True Label:", y_test)
    print("Posterior Probability:", posterior_tst)

    # 正解クラスと事後確率を保存
    y_train_post_list.extend(posterior_trn[:,[0]])
    y_train_list.extend(y_train)
    y_test_post_list.append(posterior_tst[0][0])
    y_test_list.extend(y_test)
        
# ROC曲線の描画とAUCの算出
fpr_trn, tpr_trn, thresholds_trn = roc_curve(y_train_list, y_train_post_list, pos_label=0)
roc_auc_trn = auc(fpr_trn, tpr_trn)
plt.plot(fpr_trn, tpr_trn, 'k--',label='ROC for training data (AUC = %0.2f)' % roc_auc_trn, lw=2, linestyle="-")

fpr_tst, tpr_tst, thresholds_tst = roc_curve(y_test_list, y_test_post_list, pos_label=0)
roc_auc_tst = auc(fpr_tst, tpr_tst)
plt.plot(fpr_tst, tpr_tst, 'k--',label='ROC for test data (AUC = %0.2f)' % roc_auc_tst, lw=2, linestyle="--")

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()
