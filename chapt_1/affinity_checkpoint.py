#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/3 10:23
# @Author :Kira
# @Software：PyCharm
import numpy as np
from collections import defaultdict

dataset_filename = 'affinity_dataset.txt'
X = np.loadtxt(dataset_filename)
n_samples, n_features = X.shape
print(X[:5])
features = ["bread", "milk", "cheese", "apples", 'bananas']
# 面包 牛奶 奶酪 苹果 香蕉
# 统计购买苹果数量
num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:
        num_apple_purchases += 1
print('{0} people bought apples'.format(num_apple_purchases))

rule_valid = 0
rule_invalid = 0
for sample in X:
    if sample[3] == 1:
        if sample[4] == 1:
            rule_valid += 1
        else:
            rule_invalid += 1
print("{0} cases of the rule being valid were discovered".format(rule_valid))
print("{0} cases of the rule being invalid were discovered".format(rule_invalid))
support = rule_valid
confidence = rule_valid / num_apple_purchases
print("The support is {0} and the confidence is {1}".format(support, confidence))
print("As a percentage, that is {0:.1f}%".format(confidence * 100))
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)
for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0:
            continue
        num_occurances[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:
                continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1
print(valid_rules)
print(invalid_rules)
print(num_occurances)
support = valid_rules
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)] / num_occurances[premise]
print(confidence)
for premise, conclusion in confidence:
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")


def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")

sorted_support = sorted(support.items(), key=lambda x: x[1], reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)
print("----------")

sorted_confidence = sorted(confidence.items(), key=lambda x: x[1], reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)
print("----------")
