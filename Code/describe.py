import numpy as np
import pandas as pd
import scipy.stats as stats

# 读取数据
X = pd.read_excel('5-X.xlsx', header=None).values  # 10000行，100列的矩阵
Y = pd.read_excel('5-Y.xlsx', header=None).values  # 10000行，1列的矩阵

# 检查 Y 的形状
print("Y 形状：", Y.shape)
if Y.shape[1] != 1:
    raise ValueError("Y 必须是单列向量")

# 计算X和Y的均值
mean_X = np.mean(X, axis=0)  # 对每一列计算均值
mean_Y = np.mean(Y, axis=0)  # 对Y（列）计算均值

# 计算分子部分 (X_i - mean_X) * (Y_i - mean_Y)
numerator = np.sum((X - mean_X) * (Y - mean_Y), axis=0)

# 计算分母部分 sqrt( sum((X_i - mean_X)^2) * sum((Y_i - mean_Y)^2) )
denominator_X = np.sum((X - mean_X)**2, axis=0)
denominator_Y = np.sum((Y - mean_Y)**2, axis=0)

# 检查分母是否为 0
if np.any(denominator_X == 0) or np.any(denominator_Y == 0):
    print("警告：存在方差为 0 的列，可能导致 NaN 或 inf")

# 计算皮尔逊相关系数
pearson_correlation = numerator / np.sqrt(denominator_X * denominator_Y)

# 输出计算结果
print("皮尔逊相关系数：")
print(pearson_correlation)

# 使用 SciPy 验证
pearson_scipy = np.array([stats.pearsonr(X[:, i], Y.ravel())[0] for i in range(X.shape[1])])
print("SciPy 皮尔逊相关系数：")
print(pearson_scipy)
print("是否一致：", np.allclose(pearson_correlation, pearson_scipy, rtol=1e-5, atol=1e-8, equal_nan=True))