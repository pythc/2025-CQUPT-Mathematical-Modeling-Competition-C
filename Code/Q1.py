import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, RANSACRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. 读取数据
A = pd.read_excel('A.xlsx', header=None)
B = pd.read_excel('B.xlsx', header=None).values.flatten()
n, p = A.shape

# 2. 数据分布可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2.1 每列特征直方图（示例绘制前 5 列）
for i in range(min(1, p)):
    plt.figure(figsize=(6,4))
    sns.histplot(A[i].values if i==0 else A.iloc[:, i], bins=30, kde=True)
    plt.title(f'特征 x{i} 分布')
    plt.xlabel(f'x{i} 值')
    plt.ylabel('频数')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 2.2 特征箱线图（示例前 5 列）
plt.figure(figsize=(8,4))
sns.boxplot(data=A.iloc[:, :50])
plt.title('前50列特征箱线图')
plt.xlabel('特征')
plt.ylabel('值')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 2.3 特征相关矩阵热力图（降采样显示前10特征）
corr = A.iloc[:, :100].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=False, fmt='.2f', cmap='coolwarm')
plt.title('前100特征皮尔逊相关系数矩阵')
plt.tight_layout()
plt.show()

# 3. 标准化
scaler = StandardScaler()
A_std = scaler.fit_transform(A)

# 4. OLS 回归及诊断
X_const = add_constant(A_std)
model = OLS(B, X_const).fit()
print(model.summary())

# 5. 多重共线性 (VIF)
vif_data = pd.DataFrame({
    'feature': ['const'] + [f'x{i}' for i in range(p)],
    'VIF': [variance_inflation_factor(X_const, i) for i in range(X_const.shape[1])]
})
print('\nVIF:\n', vif_data)

# 6. 异方差检验
bp_test = het_breuschpagan(model.resid, model.model.exog)
print('\nBreusch-Pagan test:', dict(zip(
    ['LM statistic','LM p-value','F-statistic','F p-value'], bp_test
)))

# 7. 交叉验证 + 正则化
alphas = np.logspace(-4, 2, 50)
ridge = RidgeCV(alphas=alphas, cv=5).fit(A_std, B)
lasso = LassoCV(alphas=alphas, cv=5, max_iter=5000).fit(A_std, B)
print(f"\nRidge alpha: {ridge.alpha_}")
print(f"Lasso alpha: {lasso.alpha_}")

# 8. 划分并用 LassoCV 模型评估
X_train, X_test, y_train, y_test = train_test_split(A_std, B, test_size=0.2, random_state=42)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
res = y_test - y_pred

# 9. 可视化拟合结果与诊断
# 9.1 实际 vs 预测\plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, s=20, alpha=0.7)
mn, mx = y_test.min(), y_test.max()
plt.plot([mn, mx], [mn, mx], '--', color='gray')
plt.xlabel('实际')
plt.ylabel('预测')
plt.title('实际 vs 预测 (LassoCV)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 9.2 残差 vs 预测\plt.figure(figsize=(6, 4))
plt.scatter(y_pred, res, s=20, alpha=0.7)
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差 vs 预测值')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 9.3 残差分布\plt.figure(figsize=(6,4))
plt.hist(res, bins=50, density=True, alpha=0.7)
plt.xlabel('残差')
plt.ylabel('密度')
plt.title('残差分布')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 10. 离群点检测 (RANSAC)
ransac = RANSACRegressor().fit(A_std, B)
mask = ransac.inlier_mask_
outliers = ~mask
plt.figure(figsize=(6,4))
plt.scatter(ransac.predict(A_std)[mask], B[mask], s=20, alpha=0.7, label='Inliers')
plt.scatter(ransac.predict(A_std)[outliers], B[outliers], s=20, alpha=0.7, color='red', label='Outliers')
plt.plot([mn,mx],[mn,mx],'--',color='gray')
plt.xlabel('预测')
plt.ylabel('实际')
plt.title('RANSAC 检测离群点')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
