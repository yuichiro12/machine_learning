import os
import struct
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression

# MNISTデータの読み込み関数
def load_mnist(path, kind='train'):

    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        
    return images, labels

# MNISTデータの読み込み
current_path = os.path.dirname(os.path.realpath(__file__))
X_train, y_train = load_mnist(current_path, kind='train')
X_test, y_test = load_mnist(current_path, kind='t10k')
    
# 学習用に最初の1000点，テスト用に最初の300点のデータを使用
X_train = X_train[:1000][:]
y_train = y_train[:1000][:]
X_test = X_test[:300][:]
y_test = y_test[:300][:]
print('#data: %d, #feature: %d (training data)' % (X_train.shape[0], X_train.shape[1]))
print('#data: %d, #feature: %d (test data)' % (X_test.shape[0], X_test.shape[1]))

# シグモイド関数の定義

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ロジスティック識別器のインスタンスの生成と学習   

lr = LogisticRegression(penalty='l1', C=1000.0, random_state=0)
lr.fit(X_train, y_train)

# 学習データおよびテストデータに対するaccuracyの算出

y_train_pred = lr.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('accuracy for training data: %.2f%%' % (acc * 100))

y_test_pred = lr.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('accuracy for test data: %.2f%%' % (acc * 100))

# 最初の25サンプルの識別結果をプロット．t: 正解クラス，p: 識別器による推測クラス

orign_img = X_test[:25][:25]
true_lab = y_test[:25][:25]
predicted_lab = y_test_pred[:25][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = orign_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, true_lab[i], predicted_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])

# 逆正則化パラメータc（λの逆数）を変化させたときの2つの特徴量に対する重みをプロット
acc_trains, acc_tests, num_nonzero =[], [], []
weights, params = [], []
for c in np.arange(-10, 7):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train, y_train)
    weights.append(lr.coef_[1])
    num_nonzero.append(np.count_nonzero(weights))
    y_train_pred = lr.predict(X_train)
    acc_trains.append(np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]*100)
    y_test_pred = lr.predict(X_test)
    acc_tests.append(np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]*100)
    params.append(10**c)

plt.figure(figsize=(21,6))
plt.subplot(131)
weights = np.array(weights)
# Feature from pixel row 15, col 10
plt.plot(params, weights[:, 402],
         label='Feature #402 (row 15, col 10)')
# Feature from pixel row 15, col 13
plt.plot(params, weights[:, 405], linestyle='--',
         label='Feature #405 (row 15, col 13)')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')

plt.subplot(132)
plt.plot(params, num_nonzero, linestyle='--',label='The number of nonzero')
plt.ylabel('the number of nonzero')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')

plt.subplot(133)
plt.plot(params, acc_trains, linestyle='--',label='accuracy for training data')
plt.plot(params, acc_tests, linestyle='--',label='accuracy for test data')
plt.ylabel('accuracy')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()


