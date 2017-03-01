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
    counts = defaultdict(int)
    for x in sequence:
            counts[x] += 1
    return counts

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