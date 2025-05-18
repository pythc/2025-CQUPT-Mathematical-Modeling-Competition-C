import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 0. 固定随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. 读入原始数据
A = pd.read_excel('A.xlsx', header=None).values        # 特征矩阵 (n_samples, n_features)
B = pd.read_excel('B.xlsx', header=None).values.flatten()  # 目标向量 (n_samples,)

# 2. 初次划分：用于噪声敏感性和最终对比
X_tr, X_te, y_tr, y_te = train_test_split(
    A, B, test_size=0.2, random_state=RANDOM_STATE
)

# 3. 噪声敏感性实验
noise_levels = [0.00, 0.01, 0.02, 0.05, 0.10]
n_repeats = 10
mse_noise_mean = []
mse_noise_std  = []

feat_std = X_tr.std(axis=0, ddof=0)
for nl in noise_levels:
    mses = []
    for rep in range(n_repeats):
        np.random.seed(RANDOM_STATE + rep)
        sigma = nl * feat_std
        noise = np.random.normal(scale=sigma, size=X_tr.shape)
        Xn_tr = X_tr + noise
        m = LinearRegression().fit(Xn_tr, y_tr)
        y_pred = m.predict(X_te)
        mses.append(mean_squared_error(y_te, y_pred))
    mse_noise_mean.append(np.mean(mses))
    mse_noise_std.append(np.std(mses))

# 4. 偏差–方差–噪声分解
# 4.1 Bias² & Variance（使用同一 X_te）
T = 200
all_preds = np.zeros((T, len(y_te)))
for i in range(T):
    Xtr_i, _, ytr_i, _ = train_test_split(A, B, test_size=0.2, random_state=i)
    m = LinearRegression().fit(Xtr_i, ytr_i)
    all_preds[i] = m.predict(X_te)
mean_pred = all_preds.mean(axis=0)
bias2    = np.mean((mean_pred - y_te) ** 2)
variance = np.mean(all_preds.var(axis=0))

# 4.2 Noise（观测噪声方差）：用高度过拟合模型拟合训练集残差
overfit = GradientBoostingRegressor(max_depth=10, n_estimators=200, random_state=RANDOM_STATE)
overfit.fit(X_tr, y_tr)
res_tr_over = y_tr - overfit.predict(X_tr)
noise_approx = np.var(res_tr_over, ddof=0)

# 4.3 验证分解
model0 = LinearRegression().fit(X_tr, y_tr)
mse0 = mean_squared_error(y_te, model0.predict(X_te))
decomp_sum = bias2 + variance + noise_approx

# 5. 异常值贡献分析（测试集残差前5%）
res_te = y_te - model0.predict(X_te)
n_out = max(int(0.05 * len(res_te)), 1)
idx_sorted = np.argsort(np.abs(res_te))[::-1]
out_idx = idx_sorted[:n_out]
mse_outliers = np.mean(res_te[out_idx] ** 2)
eps = 1e-8
mse_inliers = ((len(res_te)*mse0) - n_out*mse_outliers) / (len(res_te)-n_out + eps)

# 6. 打印结果
print("— 噪声敏感性实验 —")
print("比例    平均 MSE      标准差")
for nl, m, s in zip(noise_levels, mse_noise_mean, mse_noise_std):
    print(f"{nl:>6.2%}  {m:.4e}  ±{s:.4e}")
print()

print("— 偏差–方差–噪声分解 —")
print(f"Bias²          = {bias2:.4e}")
print(f"Variance       = {variance:.4e}")
print(f"Noise (σ²)     = {noise_approx:.4e}")
print(f"三项之和       = {decomp_sum:.4e}")
print(f"测试集 MSE     = {mse0:.4e}")
print(f"差值（Sum - MSE）= {decomp_sum - mse0:.4e}")
print()

print("— 异常值（前 5%）贡献 —")
print(f"测试集总 MSE       = {mse0:.4e}")
print(f"前 {n_out} 样本 MSE = {mse_outliers:.4e}")
print(f"其余样本 MSE      = {mse_inliers:.4e}")
print(f"贡献占比          = {n_out*mse_outliers/(len(res_te)*mse0):.2%}")

# 7. 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 噪声敏感性
plt.figure(figsize=(6,4))
plt.errorbar(np.array(noise_levels)*100, mse_noise_mean, yerr=mse_noise_std,
             marker='o', capsize=3)
plt.xlabel('训练噪声比例 (%)'); plt.ylabel('测试 MSE')
plt.title('噪声敏感性')
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 分解柱状图
labels = ['Bias²','Var','Noise']
vals   = [bias2, variance, noise_approx]
plt.figure(figsize=(6,4))
bars = plt.bar(labels, vals, alpha=0.7)
for b,v in zip(bars, vals):
    plt.text(b.get_x()+b.get_width()/2, v*1.01, f'{v:.1e}', ha='center')
plt.xlabel('成分'); plt.ylabel('MSE 贡献'); plt.title('误差分解'); plt.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.show()

# 残差箱线图
plt.figure(figsize=(6,4))
plt.boxplot(res_te, vert=False, showfliers=True)
plt.title('测试集残差箱线图'); plt.xlabel('残差'); plt.tight_layout(); plt.show()

# 异常 vs 内点
labels2 = [f'内点({len(res_te)-n_out})', f'异常({n_out})']
vals2   = [mse_inliers, mse_outliers]
plt.figure(figsize=(6,4))
bars2 = plt.bar(labels2, vals2, alpha=0.7)
for b,v in zip(bars2, vals2):
    plt.text(b.get_x()+b.get_width()/2, v*1.01, f'{v:.1e}', ha='center')
plt.title('异常 vs 内点 MSE'); plt.ylabel('MSE'); plt.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.show()

# MSE 贡献饼图
plt.figure(figsize=(6,4))
plt.pie(vals2, labels=labels2, autopct='%1.1f%%', startangle=90)
plt.title('MSE 贡献比例')
plt.tight_layout(); plt.show()
