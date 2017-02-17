#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/16 15:40
# @Author :Kira
# @Software：PyCharm
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

adult_filename = 'adult.data'
adult = pd.read_csv(adult_filename, header=None, names=["Age", "Work-Class", "fnlwgt", "Education", "Education-Num",
                                                        "Marital-Status", "Occupation", "Relationship", "Race", "Sex",
                                                        "Capital-gain", "Capital-loss", "Hours-per-week", "Native-Country",
                                                        "Earnings-Raw"])
adult.dropna(how='all', inplace=True)
adult["LongHours"] = adult["Hours-per-week"] > 40

X = adult[['Age', 'Education-Num', 'Capital-gain', 'Capital-loss', 'Hours-per-week']].values
y = (adult['Earnings-Raw'] == ' >50K').values
transformer = SelectKBest(score_func=chi2, k=3)
Xt_chi2 = transformer.fit_transform(X, y)
print(transformer.scores_)



def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:,column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))
transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
Xt_person = transformer.fit_transform(X, y)
print(transformer.scores_)

clf = DecisionTreeClassifier(random_state=14)
scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring="accuracy")
scores_pearson = cross_val_score(clf, Xt_person, y, scoring="accuracy")
print(np.mean(scores_chi2), np.mean(scores_pearson))