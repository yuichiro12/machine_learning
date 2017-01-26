import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

df_wine = pd.read_csv("wine.data", header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
'OD280/OD315 of diluted wines', 'Proline']

random_seed = 1
test_proportion = 0.5

used_feature1 = 'Alcohol'
used_feature2 = 'Malic acid'
X = df_wine[[used_feature1, used_feature2]].values
y = df_wine['Class label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_proportion, random_state = random_seed)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


plt.figure(figsize=(20,10))

x1_min, x1_max = X_train_std[:, 0].min() - 0.5, X_train_std[:, 0].max() + 0.5
x2_min, x2_max = X_train_std[:, 1].min() - 0.5, X_train_std[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))

markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])


fig_no_upper = 241
fig_no_lower = 245

# それぞれの識別に対してプロットする関数
def plot_subfigure(Z1, classifer_name):
    global fig_no_upper, fig_no_lower
    Z1 = Z1.reshape(xx1.shape)

    plt.subplot(fig_no_upper)
    fig_no_upper += 1

    plt.contourf(xx1, xx2, Z1, alpha=0.5, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y_train)):
        plt.scatter(x=X_train_std[y_train == cl, 0], y=X_train_std[y_train == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.xlabel(used_feature1+' [standardized]')
    plt.ylabel(used_feature2+' [standardized]')
    plt.title(classifer_name+'\ntrain_data')

    plt.subplot(fig_no_lower)
    fig_no_lower += 1

    plt.contourf(xx1, xx2, Z1, alpha=0.5, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y_test)):
        plt.scatter(x=X_test_std[y_test == cl, 0], y=X_test_std[y_test == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.xlabel(used_feature1+' [standardized]')
    plt.ylabel(used_feature2+' [standardized]')
    plt.title('test_data')


# Decision Tree
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3,
                              random_state=1)

tree.fit(X_train_std, y_train)
plot_subfigure(tree.fit_transform(X_test_std, y_test), "Decision Tree")

# Random Forest
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=100,
                                random_state=1,
                                n_jobs=2)

forest.fit(X_train_std, y_train)
plot_subfigure(forest.fit_transform(X_test_std, y_test), "Random Forest")

# Bagging
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=100,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)

bag.fit(X_train_std, y_train)
plot_subfigure(bag.fit_transform(X_test_std, y_test), "Bagging")

# Adaboost
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=100,
                         learning_rate=0.1,
                         random_state=1)

ada.fit(X_train_std, y_train)
plot_subfigure(ada.fit_transform(X_test_std, y_test), "Adaboost")


print('Assessing Feature Importances with Random Forests')
# TODO


plt.show()
