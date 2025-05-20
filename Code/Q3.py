import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import lasso_path, Lasso, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# 全局绘图设置
# -----------------------------------------------------------------------------
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']        # 中文字体
plt.rcParams['axes.unicode_minus'] = False         # 负号用 ASCII

# -----------------------------------------------------------------------------
# 1. 数据加载
# -----------------------------------------------------------------------------
X_raw = pd.read_excel('3-X.xlsx', header=None).values
y     = pd.read_excel('3-Y.xlsx', header=None).values.ravel()

# -----------------------------------------------------------------------------
# 2. 归一化 + 标准化
# -----------------------------------------------------------------------------
X_norm   = MinMaxScaler().fit_transform(X_raw)     # [0,1]
X_scaled = StandardScaler().fit_transform(X_norm)  # 零均值单位方差

# -----------------------------------------------------------------------------
# 3. 小波去噪（在归一化后去噪，再标准化）
# -----------------------------------------------------------------------------
def wavelet_denoise(X, wavelet='db4', level=5):
    Xden = np.zeros_like(X)
    for j in range(X.shape[1]):
        coeffs = pywt.wavedec(X[:, j], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        Xden[:, j] = pywt.waverec(coeffs, wavelet)[:X.shape[0]]
    return Xden

X_den_norm = wavelet_denoise(X_norm)
X_den      = StandardScaler().fit_transform(X_den_norm)

# -----------------------------------------------------------------------------
# 4. 相关系数对比（前20特征）
# -----------------------------------------------------------------------------
def compute_corr(X):
    return [pearsonr(X[:, j], y)[0] for j in range(X.shape[1])]

corr_raw      = compute_corr(X_raw)
corr_norm     = compute_corr(X_norm)
corr_denoised = compute_corr(X_den_norm)

k = 20
inds = np.arange(1, k+1)
w = 0.25

plt.figure(figsize=(12, 6))
plt.bar(inds - w, corr_raw[:k],      width=w, label='原始', alpha=0.8)
plt.bar(inds,      corr_norm[:k],     width=w, label='归一化', alpha=0.8)
plt.bar(inds + w,  corr_denoised[:k], width=w, label='去噪', alpha=0.8)
plt.xlabel('特征索引')
plt.ylabel('Pearson相关系数')
plt.title('前20特征 vs Y 相关系数对比')
plt.xticks(inds)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 5. 3D 点云可视化（三个示例特征）
# -----------------------------------------------------------------------------
def plot_3d(X, title):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='viridis', s=20, alpha=0.7)
    ax.set_xlabel('特征1'); ax.set_ylabel('特征2'); ax.set_zlabel('特征3')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='Y 值')
    plt.tight_layout()
    plt.show()

plot_3d(X_raw,      '原始 3D 点云')
plot_3d(X_norm,     '归一化 3D 点云')
plot_3d(X_den_norm, '去噪后 3D 点云')

# -----------------------------------------------------------------------------
# 6. Lasso 正则化路径（自动 alphas）
# -----------------------------------------------------------------------------
alphas_path, coefs, _ = lasso_path(X_den, y)
print(f"自动生成的 alphas_path 范围: {alphas_path.min():.2e} — {alphas_path.max():.2e}")

plt.figure(figsize=(8, 6))
for coef in coefs:
    plt.plot(alphas_path, coef, alpha=0.5)
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel('Alpha (log scale)')
plt.ylabel('系数值')
plt.title('Lasso 正则化路径（标准化后）')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 7. LassoCV 自动选 α & CV 曲线
# -----------------------------------------------------------------------------
lasso_cv = LassoCV(
    alphas=alphas_path,
    cv=5,
    max_iter=5000,
    random_state=42
)
lasso_cv.fit(X_den, y)

mp = lasso_cv.mse_path_
print("mse_path_ shape =", mp.shape)
# 根据 shape 决定沿哪个轴平均
if mp.shape[0] == len(lasso_cv.alphas_):
    mse_mean = mp.mean(axis=1)
else:
    mse_mean = mp.mean(axis=0)

plt.figure(figsize=(8, 6))
plt.plot(lasso_cv.alphas_, mse_mean, 'o-', label='平均 CV MSE')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.axvline(lasso_cv.alpha_, linestyle='--', color='red',
            label=f'最佳 α={lasso_cv.alpha_:.2e}')
plt.xlabel('Alpha')
plt.ylabel('平均 CV MSE')
plt.title('LassoCV 交叉验证曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 8. 最终 Lasso 拟合与诊断
# -----------------------------------------------------------------------------
model = Lasso(alpha=lasso_cv.alpha_, max_iter=5000)
model.fit(X_den, y)
y_pred = model.predict(X_den)
res    = y - y_pred

print("\nLassoCV 最终模型评估：")
print(f" 最佳 α = {lasso_cv.alpha_:.2e}")
print(f" R²  = {r2_score(y, y_pred):.4f}")
print(f" MSE = {mean_squared_error(y, y_pred):.4f}")
print(f" MAE = {mean_absolute_error(y, y_pred):.4f}")

# 实际 vs 预测
plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, s=20, alpha=0.7)
mn, mx = y.min(), y.max()
plt.plot([mn, mx], [mn, mx], '--', color='gray')
plt.xlabel('实际')
plt.ylabel('预测')
plt.title('实际 vs 预测 (LassoCV)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 残差 vs 预测
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, res, s=20, alpha=0.7)
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差 vs 预测值')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 残差分布
plt.figure(figsize=(6,4))
plt.hist(res, bins=50, density=True, alpha=0.7)
plt.xlabel('残差')
plt.ylabel('密度')
plt.title('残差分布')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
