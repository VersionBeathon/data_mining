#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/17 14:53
# @Author :Kira
# @Software：PyCharm
import os
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd

data_filename = 'ad.data'


def convert_number(x):
     try:
        return float(x)
     except ValueError:
        return np.nan

converters = {i: convert_number for i in range(1588)}
converters[1558] = lambda x: 1 if x.strip() == "ad." else 0

ads = pd.read_csv(data_filename, header=None, converters=converters)
print(ads[:5])
X = ads.drop(1558, axis=1).values
y = ads[1558]
X.astype(np.float64)


clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X, y, scoring='accuracy')
print("The average score is {:.4f}".format(np.mean(scores)))
pca = PCA(n_components=5)
Xd = pca.fit_transform(X)
np.set_printoptions(precision=3, suppress=True)
print(pca.explained_variance_ratio_)
scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')
print("The average score from the reduced dataset is {:.4f}".format(np.mean(scores_reduced)))
classes = set(y)
colors = ['red', 'green']
for cur_class, color in zip(classes, colors):
    mask = (y == cur_class).values
    plt.scatter(Xd[mask,0], Xd[mask,1], marker='o', color=color, label=int(cur_class))
plt.legend()

plt.show()