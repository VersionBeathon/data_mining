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
#print(open(path).readline())
records = [json.loads(line) for line in open(path)]
#print(records[0])
#print(records[0]['tz'])
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
#print(time_zones[:10])


def get_counts(sequence):
    _counts = defaultdict(int)
    for x in sequence:
            _counts[x] += 1
    return _counts

counts = get_counts(time_zones)
#pprint(counts)
#pprint(len(time_zones))

counts = Counter(time_zones)
#pprint(counts.most_common(10))

frame = DataFrame(records)
#print("=====================================")
#print(frame['tz'][:10])
tz_counts = frame['tz'].value_counts()
#print(tz_counts[:10])

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
# 反序
print(sorted_by_diff[::-1][:15])

# 根据电影名称分组的得分数据的标准差
rating_std_by_title = data.groupby('title')['rating'].std()
# 根据active_titles进行过滤
rating_std_by_title = ratings_by_title.ix[active_titles]
print(rating_std_by_title.sort_values(ascending=False)[:10])

# 全美婴儿姓名数据
filename = r'H:\learning\pydata-book\ch02\names\yob1880.txt'
names1880 = pd.read_csv(filename, names=['name', 'sex', 'births'])
print(names1880)
print('===================')

# 使用birth列sex分组统计当年出生情况
print(names1880.groupby('sex').births.sum())

# 将1880~2010年数据统计到一个DataFrame里面并加上year字段
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = r'H:\learning\pydata-book\ch02\names\yob{0}.txt'.format(str(year))
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
# 将所有数据整合到单个DataFrame中，ignore_index 不保留原始行号
names = pd.concat(pieces, ignore_index=True)
print(names)
# 对数据进行聚合
total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
print(total_births.tail())
print(names.groupby('sex').births.sum())
total_births.plot(title='Total births by sex and year')
plt.show()


# 指定名字的婴儿数相对于总出生数的比例
def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)
print(names)

# 验证所有分组的prop和是否为1
print(np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1))


# 每对sex/year组合的前1000个名字进行分组
def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
print(top1000[:1000])

# 分成男女两个数据集
boys  = top1000[top1000.sex == "M"]
girls = top1000[top1000.sex == "F"]

# 生成按year和name统计的总出生数透视表
total_births = top1000.pivot_table('births', index='year', columns='name', aggfunc=sum)

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")
plt.show()

# 按year和sex进行聚合并绘图
table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
table.plot(title='Sum of table 1000.prop by year and sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
plt.show()

# 只考虑2010年男孩的名字
df = boys[boys.year == 2010]
print(df)

# 通过prop的累计和cumsum，通过searchsorted找出累加到0.5的人数
prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
print(prop_cumsum[:10])
print(prop_cumsum.searchsorted(0.5) + 1)

df = boys[boys.year == 1900]
in1900 = df.sort_values(by='prop', ascending=False).prop.cumsum()
print(in1900.searchsorted(0.5) + 1)


def get_quantitle_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantitle_count)
diversity = diversity.unstack('sex')
print(diversity.head())

# 从names列取出最后一个字母
get_last_letter = lambda  x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('births', index=last_letters, columns=['sex', 'year'], aggfunc=sum)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
print(subtable.head())

# 计算各性别各末字母占总出生人数的比例
print(subtable.sum())
letter_prop = subtable / subtable.sum().astype(float)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title="Male")
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
plt.show()


# 绘制名字趋势图
letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y'],'M'].T
print(dny_ts.head())
dny_ts.plot()
plt.show()

# 变成女孩名字的男孩名字（以及相反的情况）
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
print(lesley_like)
# 利用这个结果过滤其他的名字，并按名字分组计算出生数以查看相对频率
filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.groupby('name').births.sum())
# 按性别和年度进行聚合，并按年度进行规范化处理
table = filtered.pivot_table('births', index='year', columns='sex', aggfunc=sum)
table = table.div(table.sum(1), axis=0)
print(table.tail())
table.plot(style={'M': 'k-', 'F': 'k--'})
plt.show()