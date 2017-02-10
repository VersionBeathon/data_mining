#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/9 14:58
# @Author :Kira
# @Software：PyCharm
import pandas as pd
from collections import defaultdict
from operator import itemgetter

rating_filename = 'ml-1m/ratings.dat'
all_ratings = pd.read_csv(rating_filename, delimiter="::", header=None, names=["UserID", "MovieID", "Rating", "Datetime"])
all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"], unit="s")
print(all_ratings[:5])
all_ratings["Favorable"] = all_ratings["Rating"] > 3
print(all_ratings[10: 15])
ratings = all_ratings[all_ratings['UserID'].isin(range(200))]
favorable_ratings = ratings[ratings["Favorable"]]
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings[0: 10000].groupby("UserID")["MovieID"])
num_favorable_by_movie = ratings[["MovieID", "Favorable"]].groupby("MovieID").sum()
print(num_favorable_by_movie.sort_values(by="Favorable", ascending=False)[:5])

frequent_itemsets = {}
min_support = 50
frequent_itemsets[1] = dict((frozenset((movie_id,)), row["Favorable"]) for movie_id, row in num_favorable_by_movie.iterrows() if row["Favorable"] > min_support)


def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie, ))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])

for k in range(2, 20):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1], min_support)
    frequent_itemsets[k] = cur_frequent_itemsets
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent itemsets of length {0}".format(k))
        break
    else:
        print("I found {0} frequent itemsets of length {1}".format(len(cur_frequent_itemsets), k))
del frequent_itemsets[1]

candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print(candidate_rules)
correct_counts = defaultdict(int)
incorrenct_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrenct_counts[candidate_rule] += 1

rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrenct_counts[candidate_rule]) for candidate_rule in candidate_rules}
sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print(" ")