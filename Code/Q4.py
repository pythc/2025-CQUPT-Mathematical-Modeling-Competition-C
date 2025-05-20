import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from time import time

# 数据加载、预处理
X = pd.read_excel('4-X.xlsx', header=None).to_numpy()
y = pd.read_excel('4-Y.xlsx', header=None).squeeze().to_numpy()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 手动加一列全 1（相当于截距）
X_design = np.concatenate([np.ones((X_scaled.shape[0],1)), X_scaled], axis=1)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_design, y, test_size=0.2, random_state=42
)

# 网格搜索
C_list   = np.logspace(-3, 3, num=10)
eps_list = np.logspace(-3, 1, num=10)

def fit_and_score(C_val, eps_val, Xdata, ydata):
    t_start = time()
    svr = SVR(kernel='linear', C=C_val, epsilon=eps_val)
    svr.fit(Xdata, ydata)
    y_hat = svr.predict(Xdata)
    score = r2_score(ydata, y_hat)
    t_used = time() - t_start
    print(f"C={C_val:.4g}, ε={eps_val:.4g} ⇒ R²={score:.4f} （耗时 {t_used:.2f}s）")
    return {
        'C': C_val, 'epsilon': eps_val,
        'logC': np.log10(C_val), 'logEps': np.log10(eps_val),
        'R2': score, 'time': t_used
    }

all_results = Parallel(n_jobs=-1)(
    delayed(fit_and_score)(C, eps, X_tr, y_tr)
    for C in C_list for eps in eps_list
)
df = pd.DataFrame(all_results)

# 挑最佳 看趋势
best_row = df.loc[df['R2'].idxmax()]
best_C, best_eps = best_row['C'], best_row['epsilon']
print("\n最佳参数：")
print(best_row[['C','epsilon','R2']])

print("\nTop5 R2 排行：")
print(df.nlargest(5, 'R2')[['C','epsilon','R2']])

print("\nlogC/logEps 与 R2 的相关性：")
print(df[['logC','logEps','R2']].corr())

# 用最优参数跑一次，输出训练/测试指标
svr_best = SVR(kernel='linear', C=best_C, epsilon=best_eps)
t0 = time()
svr_best.fit(X_tr, y_tr)
print(f"耗时{time() - t0:.2f}s")

# 训练集评估
y_tr_pred = svr_best.predict(X_tr)
r2_tr = r2_score(y_tr, y_tr_pred)
mse_tr = mean_squared_error(y_tr, y_tr_pred)
mae_tr = mean_absolute_error(y_tr, y_tr_pred)
rmse_tr = np.sqrt(mse_tr)

# 测试集评估
y_te_pred = svr_best.predict(X_te)
r2_te = r2_score(y_te, y_te_pred)
mse_te = mean_squared_error(y_te, y_te_pred)
mae_te = mean_absolute_error(y_te, y_te_pred)
rmse_te = np.sqrt(mse_te)

print("\n模型评估——")
print(f"  · 训练集 R²={r2_tr:.4f}, MSE={mse_tr:.4f}, MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}")
print(f"  · 测试集 R²={r2_te:.4f}, MSE={mse_te:.4f}, MAE={mae_te:.4f}, RMSE={rmse_te:.4f}")

# 交叉验证看看稳定性
cv_mse = -cross_val_score(
    svr_best, X_tr, y_tr, cv=5, scoring='neg_mean_squared_error'
)
print(f"CV MSE平均={cv_mse.mean():.4f}, 方差={cv_mse.std():.4f}")

# 支持向量数+残差分布
n_support = len(svr_best.support_)
print(f"\n模型用{n_support}个支持向量，共{X_tr.shape[0]}训练样本")
resid = y_te - y_te_pred
print("测试集残差统计：",
      f"最大 {resid.max():.4f}, 最小 {resid.min():.4f}, σ={resid.std():.4f}")

plt.figure(figsize=(12,4))

# log10(C) vs R2
plt.subplot(1,3,1)
sorted_by_C = df.sort_values('logC')
plt.plot(sorted_by_C['logC'], sorted_by_C['R2'], marker='o')
plt.xlabel('log10(C)')
plt.ylabel('R2')
plt.title('log10(C) vs R2')

# log10(epsilon) vs R2
plt.subplot(1,3,2)
sorted_by_eps = df.sort_values('logEps')
plt.plot(sorted_by_eps['logEps'], sorted_by_eps['R2'], marker='o')
plt.xlabel('log10(epsilon)')
plt.ylabel('R2')
plt.title('log10(epsilon) vs R2')

# R2热力图
plt.subplot(1,3,3)
heat_data = df.pivot(index='logEps', columns='logC', values='R2')
plt.imshow(heat_data, origin='lower', aspect='auto')
plt.colorbar(label='R2')
plt.xlabel('log10(C)')
plt.ylabel('log10(epsilon)')
plt.title('R2 Heatmap')

plt.tight_layout()
plt.show()

