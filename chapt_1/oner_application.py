#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/3 16:02
# @Author :Kira
# @Software：PyCharm
from sklearn.datasets import load_iris
from collections import defaultdict
from operator import itemgetter
from sklearn.cross_validation import train_test_split
import numpy as np

dataset = load_iris()
# 数据集
X = dataset.data
# 类别信息
y = dataset.target
print(y, len(y))
'''
特征值为连续性，无数个可能的值，连续值得另一个特点，两个值相近，表示相似度很大
类别的取值为离散型， 0 1 2 代表三个类别
数据集的特征为连续值，即将使用的算法使用类别型特征值，
需要将连续值转变为类别型，这个过程叫作离散化（设定阈值）
'''
# 设定阈值（特征值的平均数）
attribute_means = X.mean(axis=0)
print(attribute_means)
# 将数据集打散， 把连续的特征值转换为类别型
X_d = np.array( X >= attribute_means, dtype="int")
print(X_d)
# 将数据分为训练集和测试集
Xd_train, Xd_test, y_train, y_test = train_test_split(X_d, y, random_state=14)


def train_feature_value(X, y_true, feature_index, value):
    '''
    :param X:数据集
    :param y_true:类别数组
    :param feature_index:选好的特征索引值
    :param value: 特征值
    :return:待预测的个体类别和错误率
    '''
    class_counts = defaultdict(int)
    # 统计具有给定特征值的个体在每个类别中出现次数
    for sample, y in zip(X, y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    # 计算错误率（具有该特征值的个体在其他类别中出现次数）
    error = sum([class_count for class_value, class_count in class_counts.items() if class_value != most_frequent_class])
    return most_frequent_class, error


def train_on_feature(X, y_true, feature_index):
    '''
    :param X: 数据集
    :param y_true: 类别数组
    :param feature_index: 选好的特征索引值
    :return:预测器， 错误率
    '''
    values = set(X[:,feature_index])
    # key为特征值， value为类别
    predictors = {}
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature_index, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error


all_predictors = {variable: train_on_feature(Xd_train, y_train, variable) for variable in range(Xd_train.shape[1])}
print(all_predictors)
errors = {variable: error for variable, (mapping, error) in all_predictors.items()}
best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))

model = {'variable': best_variable,
         'predictor': all_predictors[best_variable][0]}
print(model)


def predict(Xd_test, model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in Xd_test])
    return y_predicted

y_predicted = predict(Xd_test, model)
print(y_predicted)
accuracy = np.mean(y_predicted == y_test) * 100
print("The test accuracy is {:.1f}%".format(accuracy))