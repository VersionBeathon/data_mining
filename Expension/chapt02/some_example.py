#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/3/1 16:14
# @Author :Kira
# @Software：PyCharm
import json
from collections import defaultdict
from pprint import pprint
from collections import Counter
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
path = r'H:\learning\pydata-book\ch02\usagov_bitly_data2012-03-16-1331923249.txt'
print(open(path).readline())
records = [json.loads(line) for line in open(path)]
print(records[0])
print(records[0]['tz'])
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
print(time_zones[:10])


def get_counts(sequence):
    _counts = defaultdict(int)
    for x in sequence:
            _counts[x] += 1
    return _counts

counts = get_counts(time_zones)
pprint(counts)
pprint(len(time_zones))

counts = Counter(time_zones)
pprint(counts.most_common(10))

frame = DataFrame(records)
print(frame)
print("=====================================")
print(frame['tz'][:10])
tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])

print("=====================================")
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])
tz_counts[:10].plot(kind='barh', rot=0)
plt.show()

results = Series([x.split()[0] for x in frame.a.dropna()])
print(results[:5])
print(results.value_counts()[:8])

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
print(operating_system[:5])

by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])

indexer = agg_counts.sum(1).argsort()
print(indexer[:10])

count_subset = agg_counts.take(indexer)[-10:]
print(count_subset)
count_subset.plot(kind='barh', stacked=True)
plt.show()
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
plt.show()

# 影评数据
user_path = r'H:\learning\pydata-book\ch02\movielens\users.dat'
rate_path = r'H:\learning\pydata-book\ch02\movielens\ratings.dat'
movie_path = r'H:\learning\pydata-book\ch02\movielens\movies.dat'
unames = ['user_id', 'gender', 'age', 'occupations', 'zip']
users = pd.read_table(user_path, sep='::', header=None, names=unames)
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(rate_path, sep='::', header=None, names=rnames)
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(movie_path, sep='::', header=None, names=mnames)
print(users[:5])
print(ratings[:5])
print(movies[:5])

# 合并操作
data = pd.merge(pd.merge(ratings, users), movies)
print(data.ix[0])

# 按性别计算每部电影平均得分
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
print(mean_ratings[:5])
print("================================")

# 过滤掉评分数据不够250条的电影
# 对title 进行分组，利用size得到一个含有各电影分组大小的Series
ratings_by_title = data.groupby('title').size()
print(ratings_by_title[:10])
active_titles = ratings_by_title.index[ratings_by_title >= 250]
print(active_titles)
mean_ratings = mean_ratings.ix[active_titles]
print(mean_ratings)

# 了解女性观众最喜欢的电影
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)
print(top_female_ratings[:10])

# 计算男女分歧
mean_ratings['diff'] = mean_ratings["M"] - mean_ratings["F"]
sorted_by_diff = mean_ratings.sort_values(by='diff')
print(sorted_by_diff[:15])