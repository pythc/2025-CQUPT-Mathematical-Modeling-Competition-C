import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 读取数据
X = pd.read_excel('3-X.xlsx', header=None).values
Y = pd.read_excel('3-Y.xlsx', header=None).values.ravel()

# 归一化
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# 小波去噪db4且level=5
import pywt

def wavelet_denoise(X, wavelet='db4', level=5):
    X_denoised = np.zeros_like(X)
    for i in range(X.shape[1]):
        coeffs = pywt.wavedec(X[:, i], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])  # 去掉低频部分（逼近系数）
        rec = pywt.waverec(coeffs, wavelet)
        X_denoised[:, i] = rec[:X.shape[0]]
    return X_denoised

X_denoised = wavelet_denoise(X_norm, wavelet='db4')

# 训练随机森林回归模型并进行超参数调优
# 去掉常数项，直接使用去噪后的数据
X_input = X_denoised

# 随机森林参数调优
param_grid = {
    'n_estimators': [50, 100, 200],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 分裂内部节点时所需的最小样本数
    'min_samples_leaf': [1, 2, 4]  # 每个叶子节点的最小样本数
}

# 初始化模型
rf_model = RandomForestRegressor(random_state=42)

# 使用GridSearchCV进行超参数优化
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_input, Y)

# 输出最佳超参数
print("最佳超参数：", grid_search.best_params_)

# 使用最佳超参数训练最终模型
best_rf_model = grid_search.best_estimator_

# 预测
Y_pred_rf = best_rf_model.predict(X_input)

# 评估模型
r2_rf = r2_score(Y, Y_pred_rf)
mse_rf = mean_squared_error(Y, Y_pred_rf)
mae_rf = mean_absolute_error(Y, Y_pred_rf)
rmse_rf = np.sqrt(mse_rf)  # 计算RMSE

# 输出评估结果
print(f"随机森林回归模型：")
print(f"R² = {r2_rf:.4f}")
print(f"MSE = {mse_rf:.4f}")
print(f"MAE = {mae_rf:.4f}")
print(f"RMSE = {rmse_rf:.4f}")

# 可视化结果
plt.figure(figsize=(12, 8))

# 实际值 vs 预测值
plt.subplot(2, 2, 1)
plt.scatter(Y, Y_pred_rf, s=10)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
plt.xlabel("实际值")
plt.ylabel("预测值")
plt.title("实际值 vs 预测值（随机森林）")
plt.grid(True)

# 残差图
residuals_rf = Y - Y_pred_rf
plt.subplot(2, 2, 2)
plt.scatter(Y_pred_rf, residuals_rf, s=10, color='green')
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("预测值")
plt.ylabel("残差")
plt.title("残差 vs 预测值（随机森林）")
plt.grid(True)

# 残差分布直方图
plt.subplot(2, 2, 3)
plt.hist(residuals_rf, bins=50, density=True, alpha=0.7)
plt.xlabel("残差")
plt.ylabel("密度")
plt.title("残差分布（随机森林）")
plt.grid(True)

plt.tight_layout()
plt.show()
