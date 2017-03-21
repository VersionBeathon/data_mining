#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/3/7 21:33
# @Author :Kira
# @Software：PyCharm
import numpy as np

# ndarray
data_1 = [6, 7.5, 8, 0, 1]
arr_1 = np.array(data_1)
data_2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr_2 = np.array(data_2)
print(arr_1)
print(arr_2)
print(arr_2.ndim)
print(arr_2.shape)
print(arr_2.T)
print(arr_1.dtype)

# 使用zeros以及ones创建数组
arr_3 = np.zeros(10)
arr_4 = np.zeros((3, 6))
arr_5 = np.empty((2, 3, 2))
# arange
arr_6 = np.arange(15)
print(arr_3)
print(arr_4)
print(arr_5)
print(arr_6)
# eye、indentity 创建N*N的单位矩阵，对角线为1，其余为0（单位矩阵）
arr_7 = np.eye(4)
arr_8 = np.identity(4, dtype='float64')
print(arr_7)
print(arr_8)
# dtype创建初始类型 astype转换数据类型(非原址操作)
arr_9 = np.array([1, 2, 3], dtype=np.int32)
new_arr_9 = arr_9.astype(np.float64)
print(new_arr_9.dtype)
# 数组运算
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr)
print(arr * arr)
print(arr - arr)
print(1 / arr)
print(arr ** 0.5)
# 索引切片
arr = np.arange(10)
print(arr[5:8])
arr[5:8] = 12
print(arr)
# 修改复制数据时同时会修改数据源,使用.copy()来实现复制
arr_slice = arr[5:8]
print(arr)
arr_slice[:] = 64
print(arr)
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 多维数组
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d)
print(arr3d[0])
# 多维数组切片
print(arr2d)
print(arr2d[:2])
print(arr2d[:2, 1:])
print(arr2d[:, :1])
# bool型索引
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(data)
print(names == 'Bob')
print(data[names == 'Bob'])
print(data[names == 'Bob', 2:])
print(data[-(names == 'Bob')])
# & | 与或操作
mask = (names == 'Bob') | (names == 'Will')
print(data[mask])
print(data[data < 0])
# 通过一维布尔数组设置整行或整列的值
data[names!= 'Joe'] = 7
print(data)
# 花式索引（Fancy indexing）
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print(arr)
# 通过数组选取元素
print(arr[[4, 2, 1 ,0]])

# 选取指定行 指定列的数据
arr = np.arange(32).reshape((8, 4))
print(arr)
print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]])
print(arr[np.ix_([1, 5, 7, 2],[0, 3, 1, 2])])