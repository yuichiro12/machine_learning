import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP
import sys

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


n_training_data = 1000
n_validation_data = 300
n_test_data = 300

# my code
X_trn = X_train[:1000]
y_trn = y_train[:1000]
X_vld = X_train[1000:1300]
y_vld = y_train[1000:1300]
X_tst = X_test[:300]
y_tst = y_test[:300]
# end my code

# 多層パーセプトロン(MLP)のインスタンスの生成と学習

def mlp(ALPHA, ETA, DEC):
    print("alpha = " + str(ALPHA) + "\n" + "eta = " + str(ETA) + "\n" + "decrease_const = " + str(DEC))
    nn = NeuralNetMLP(n_output=10,                # 出力ユニット数
                      n_features=X_trn.shape[1],  # 入力ユニット数
                      n_hidden=30,                # 隠れユニット数
                      l2=0.5,                     # L2正則化のλパラメータ
                      l1=0.5,                     # L1正則化のλパラメータ
                      epochs=600,                 # 学習エポック数
                      eta=ETA,                  # 学習率の初期値
                      alpha = ALPHA,              # モーメンタム学習の1つ前の勾配の係数
                      decrease_const=DEC,     # 適応学習率の減少定数
                      minibatches=10,             # 各エポックでのミニバッチ数
                      shuffle=True,               # データのシャッフル
                      random_state=3)             # 乱数シードの状態

    nn.fit(X_trn, y_trn, print_progress=True)

    plt.figure(0)
    plt.plot(range(len(nn.cost_)), nn.cost_)
    plt.ylim([0, 1000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * minibatches')
    plt.tight_layout()


    print("")
    trn_prd = nn.predict(X_trn)
    vld_prd = nn.predict(X_vld)
    tst_prd = nn.predict(X_tst)
    cnt = 0
    for i, val in enumerate(y_trn):
        if val == trn_prd[i]:
            cnt += 1
    print(str(cnt/len(trn_prd) * 100) + "%")
    cnt = 0
    for i, val in enumerate(y_vld):
        if val == vld_prd[i]:
            cnt += 1
    print(str(cnt/len(vld_prd) * 100) + "%")
    cnt = 0
    for i, val in enumerate(y_tst):
        if val == tst_prd[i]:
            cnt += 1
    print(str(cnt/len(tst_prd) * 100) + "%")
    print("\n")

    plt.show()

ALPHA = [0.0001, 0.001, 0.01]
ETA = [0.0001, 0.001, 0.01]
DEC = [0.0001, 0.00001, 0.000001]

for i, v in enumerate(ALPHA):
    mlp(v, ETA[1], DEC[1])

for i, v in enumerate(ETA):
    mlp(ALPHA[1], v, DEC[1])

for i, v in enumerate(DEC):
    mlp(ALPHA[1], ETA[1], v)
