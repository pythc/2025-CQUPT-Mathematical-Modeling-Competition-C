# -*- coding: utf-8 -*-
"""
降维方法比较：PCA vs AE（联合训练）
回归模型：Linear, Ridge, Lasso, SVR, MLP, RandomForest
评估：重构误差 + 回归误差 + 散点图分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def build_joint_autoencoder(input_dim: int, bottleneck_dim: int) -> Model:
    inp = Input(shape=(input_dim,))
    encoded = Dense(bottleneck_dim, activation='relu',
                    activity_regularizer=regularizers.l1(1e-5))(inp)
    decoded = Dense(input_dim)(encoded)
    pred_y = Dense(1)(encoded)

    model = Model(inputs=inp, outputs=[decoded, pred_y])
    model.compile(optimizer='adam',
                  loss=['mse', 'mse'],
                  loss_weights=[0.5, 0.5])
    return model


def evaluate_k(k: int, X_std: np.ndarray, X_orig: np.ndarray, Y_std: np.ndarray, scaler_y, regressors: dict) -> dict:
    start = time.time()
    input_dim = X_std.shape[1]

    # ---- PCA 降维与重构 ----
    pca = PCA(n_components=k)
    Z = pca.fit_transform(X_std)
    Xp_std = pca.inverse_transform(Z)
    Xp = scaler.inverse_transform(Xp_std)
    pca_recon_mse = mean_squared_error(X_orig, Xp)

    pca_metrics = {}
    for name, mdl in regressors.items():
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = {'R2': [], 'MSE': [], 'MAE': []}
        for tr, te in cv.split(Xp):
            mdl.fit(Xp[tr], Y_std[tr])
            pred = scaler_y.inverse_transform(mdl.predict(Xp[te]).reshape(-1, 1)).ravel()
            y_true = scaler_y.inverse_transform(Y_std[te].reshape(-1, 1)).ravel()
            scores['R2'].append(r2_score(y_true, pred))
            scores['MSE'].append(mean_squared_error(y_true, pred))
            scores['MAE'].append(mean_absolute_error(y_true, pred))
        pca_metrics[name] = {k: np.mean(v) for k, v in scores.items()}

    # ---- AE 联合训练 ----
    ae = build_joint_autoencoder(input_dim, k)
    ae.fit(X_std, [X_std, Y_std],
           epochs=20,
           batch_size=256,
           shuffle=True,
           validation_split=0.1,
           verbose=0)

    Xa_std, pred_y_std = ae.predict(X_std)
    Xa = scaler.inverse_transform(Xa_std)
    ae_recon_mse = mean_squared_error(X_orig, Xa)

    ae_metrics = {}
    for name, mdl in regressors.items():
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = {'R2': [], 'MSE': [], 'MAE': []}
        for tr, te in cv.split(Xa):
            mdl.fit(Xa[tr], Y_std[tr])
            pred = scaler_y.inverse_transform(mdl.predict(Xa[te]).reshape(-1, 1)).ravel()
            y_true = scaler_y.inverse_transform(Y_std[te].reshape(-1, 1)).ravel()
            scores['R2'].append(r2_score(y_true, pred))
            scores['MSE'].append(mean_squared_error(y_true, pred))
            scores['MAE'].append(mean_absolute_error(y_true, pred))
        ae_metrics[name] = {k: np.mean(v) for k, v in scores.items()}

    logging.info(
        f"k={k:3d} 完成 | PCA_MSE={pca_recon_mse:.4f} | AE_R2_Linear={ae_metrics['Linear']['R2']:.4f} | 用时={time.time() - start:.2f}s")

    return {
        'k': k,
        'pca_recon_mse': pca_recon_mse,
        'ae_recon_mse': ae_recon_mse,
        'pca_metrics': pca_metrics,
        'ae_metrics': ae_metrics
    }


if __name__ == "__main__":
    # === 数据读取 ===
    X_orig = pd.read_excel('5-X.xlsx', header=None).values
    Y = pd.read_excel('5-Y.xlsx', header=None).values.ravel()

    scaler = StandardScaler()
    scaler_y = StandardScaler()

    X_std = scaler.fit_transform(X_orig)
    Y_std = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()

    # === 回归模型 ===
    regressors = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1e-2, max_iter=5000),
        'SVR': SVR(kernel='rbf', C=10, epsilon=0.1),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42),
        'RF': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # === 多维度 k 并行评估 ===
    k_list = list(range(5, 101, 5))
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_k)(k, X_std, X_orig, Y_std, scaler_y, regressors) for k in k_list
    )

    # === 汇总结果 ===
    recon_mse = [r['pca_recon_mse'] for r in results]
    ae_recon_mse = [r['ae_recon_mse'] for r in results]
    pca_r2 = {m: [r['pca_metrics'][m]['R2'] for r in results] for m in regressors}
    ae_r2 = {m: [r['ae_metrics'][m]['R2'] for r in results] for m in regressors}

    # === 绘图：重构误差 & R2 对比 ===
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(k_list, recon_mse, 'o-', label='PCA 重构 MSE')
    axes[0].plot(k_list, ae_recon_mse, 's--', label='AE 重构 MSE')
    axes[0].set_ylabel("重构 MSE")
    axes[0].set_title("重构误差 vs k")
    axes[0].legend()
    axes[0].grid(True)

    for name in regressors:
        axes[1].plot(k_list, pca_r2[name], '-o', label=f'PCA+{name}')
        axes[1].plot(k_list, ae_r2[name], '--s', label=f'AE+{name}')
    axes[1].set_xlabel("瓶颈/主成分数 k")
    axes[1].set_ylabel("测试 R2")
    axes[1].set_title("PCA vs AE + 各模型 R2 比较")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # === 绘制误差散点图：AE 重构误差 vs 回归误差 ===
    for name in regressors:
        ae_recons = np.array([r['ae_recon_mse'] for r in results])
        ae_reg_mse = np.array([r['ae_metrics'][name]['MSE'] for r in results])

        plt.figure(figsize=(6, 5))
        plt.scatter(ae_recons, ae_reg_mse, c='blue', alpha=0.7)
        plt.xlabel("AE 重构 MSE")
        plt.ylabel(f"{name} 回归 MSE")
        plt.title(f"AE 重构误差 vs 回归误差（{name}）")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === 输出最佳结果 ===
    best_idx = np.argmax(pca_r2['Linear'])
    best_k = k_list[best_idx]
    best_res = results[best_idx]

    print(f"\n>>> 最优 k (PCA+Linear) = {best_k}\n")
    print("模型       | 方法 |   R²    |   MSE    |   MAE")
    print("-----------|------|---------|----------|---------")
    for method, metrics in [('PCA', best_res['pca_metrics']),
                            ('AE', best_res['ae_metrics'])]:
        for name, vals in metrics.items():
            print(f"{name:9s} | {method:3s} | {vals['R2']:.4f} | {vals['MSE']:.4f} | {vals['MAE']:.4f}")
