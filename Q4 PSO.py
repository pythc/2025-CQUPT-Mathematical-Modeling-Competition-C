import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
X = pd.read_excel('4-X.xlsx', header=None).to_numpy()
y = pd.read_excel('4-Y.xlsx', header=None).squeeze().to_numpy()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_design = np.concatenate([np.ones((X_scaled.shape[0],1)), X_scaled], axis=1)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_design, y, test_size=0.2, random_state=42
)

# 2. 定义 PSO 的目标函数：交叉验证 MSE（越低越好）
def svr_cv_mse(params):
    """
    params: swarm_size x 2 维度数组，params[:,0]=log10(C)，params[:,1]=log10(epsilon)
    返回：每个粒子的 CV MSE（数组）
    """
    n_particles = params.shape[0]
    mse_vals = np.zeros(n_particles)
    for i in range(n_particles):
        logC, logEps = params[i]
        C_val = 10 ** logC
        eps_val = 10 ** logEps
        model = SVR(kernel='linear', C=C_val, epsilon=eps_val)
        # 这里用 3 折 CV，加快速度，如果数据量大可以改为 5 或更多
        scores = cross_val_score(model, X_tr, y_tr, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        mse_vals[i] = -scores.mean()
    return mse_vals

# 3. 配置 PSO：在 log10 空间中搜索
#    搜索范围：C ∈ [1e-3, 1e3] → log10(C) ∈ [-3, 3]
#              ε ∈ [1e-3, 1e1] → log10(ε) ∈ [-3, 1]
bounds = (np.array([-3, -3]), np.array([3,  1]))

# 4. 运行 PSO
optimizer = GlobalBestPSO(
    n_particles=30,    # 粒子数量
    dimensions=2,      # 两个维度：logC, logEps
    options={
        'c1': 0.5,     # 粒子向个体最优学习因子
        'c2': 0.3,     # 粒子向全局最优学习因子
        'w' : 0.9      # 惯性权重
    },
    bounds=bounds
)

cost, pos = optimizer.optimize(svr_cv_mse, iters=50)
best_logC, best_logEps = pos
best_C, best_eps = 10**best_logC, 10**best_logEps
print(f"PSO 优化完成 → 最优 C={best_C:.4g}, ε={best_eps:.4g}, CV MSE={cost:.4f}")

# 5. 用最优超参数训练最终模型并评估
svr_best = SVR(kernel='linear', C=best_C, epsilon=best_eps)
svr_best.fit(X_tr, y_tr)

# 训练/测试评估
def evaluate(model, X, y, name="数据集"):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"{name} MSE = {mse:.4f}")

evaluate(svr_best, X_tr, y_tr, "训练集")
evaluate(svr_best, X_te, y_te, "测试集")

# 6. 可视化：PSO 迭代过程中成本函数的收敛
plot_cost_history(optimizer.cost_history)
plt.title("PSO 收敛曲线")
plt.xlabel("迭代次数")
plt.ylabel("CV MSE")
plt.show()
