import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
from sklearn import preprocessing
from som import som


df_glass = pd.read_csv("glass.data", header=None)
df_glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

input_data = df_glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values
n_feature = input_data.shape[1]
n_class = max(df_glass['Type'].values)

sc = preprocessing.StandardScaler()
input_data = sc.fit_transform(input_data)

dim_x = 5
dim_y = 4

# インスタンス生成
s = som(x=dim_x,
        y=dim_y,
        input_len=n_feature)

# 学習
s.random_weights_init(input_data)
s.train_batch(input_data, 1000)


# 各ニューロンノードの参照ベクトルを表す棒グラフ描画
X = np.arange(len(s.weights[0][0]))
source_str = '0123456789abcdef'
color_map = [['#' + ''.join([random.choice(source_str) for x in range(6)])
              for y in range(dim_y)] for x in range(dim_x)]

plt.figure(1, figsize=(14,10))
for i in range(dim_x):
    for j in range(dim_y):
        plt.subplot2grid((dim_x,dim_y), (i, j))
        plt.bar(X, s.weights[i][j],
                facecolor=color_map[i][j],
                edgecolor='white',
                align = "center")
        plt.ylim(-3, 3)
        plt.xticks(X, ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])


# 各ニューロンに分類されたデータのクラス分布描画

dim = math.ceil(math.sqrt(n_class))
classes = df_glass['Type']
counters = [[[0 for y in range(dim_y)] for x in range(dim_x)] for c in range(n_class)]
for d, c in zip(input_data, classes):
    win = s.winner(d)
    counters[c-1][win[0]][win[1]] += 1

plt.figure(2, figsize=(14,10))
for c in range(n_class):
    plt.subplot2grid((dim, dim), (c//dim, c%dim))
    plt.imshow(counters[c], aspect='auto', interpolation='none', cmap="Blues")
    plt.title('Type'+str(c+1))
    plt.xticks(())
    plt.yticks(())
    plt.colorbar()


# クラスと特徴量の関係の積み上げ棒グラフ描画

plt.figure(3, figsize=(14,10))
for c in range(n_class):
    plt.subplot2grid((dim, dim), (c//dim, c%dim))
    sum_h_p = np.zeros(len(s.weights[0][0]))
    sum_h_n = np.zeros(len(s.weights[0][0]))
    for i in range(dim_x):
        for j in range(dim_y):
            h = s.weights[i][j]*counters[c][i][j]
            h_p = np.array([val if val>0 else 0 for val in h])
            h_n = np.array([val if val<0 else 0 for val in h])
            plt.bar(X, h_p,
                    bottom=sum_h_p,
                    facecolor=color_map[i][j],
                    edgecolor='white',
                    align='center')
            plt.bar(X, h_n,
                    bottom=sum_h_n,
                    facecolor=color_map[i][j],
                    edgecolor='white',
                    align='center')
            sum_h_p += h_p
            sum_h_n += h_n
    plt.xticks(X, ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
    plt.title('Type'+str(c+1))


# 平均の棒グラフ描画

plt.figure(4, figsize=(14,10))
for c in range(n_class):
    elm_num = 0.0
    elm_val = np.zeros(len(s.weights[0][0]))
    plt.subplot2grid((dim, dim), (c//dim, c%dim))
    for i in range(dim_x):
        for j in range(dim_y):
            elm_num += counters[c][i][j]
            elm_val += s.weights[i][j]*counters[c][i][j]
            Y = elm_val/elm_num if elm_num != 0 else np.zeros(len(s.weights[0][0]))
    plt.bar(X, Y, facecolor='#9999ff', edgecolor='white', align='center')
    plt.xticks(X, ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
    plt.ylim(-1.5, 1.5)
    plt.title('Type'+str(c+1))

plt.show()
