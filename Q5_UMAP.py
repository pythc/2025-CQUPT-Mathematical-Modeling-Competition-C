import logging
import pandas as pd
import numpy as np
import umap
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, Model, optimizers, losses, callbacks
import matplotlib.pyplot as plt
import seaborn as sns

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

# 1. 数据加载
X = pd.read_excel('5-X.xlsx', header=None).values  # 替换为实际路径
Y = pd.read_excel('5-Y.xlsx', header=None).values.ravel()

# 2. 划分 UMAP 降维训练集和重构/回归训练集
X_emb_train, X_emb_test, Y_emb_train, Y_emb_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 超参数设置
k_list = [2, 5, 10, 20, 50]
results = []

for k in k_list:
    logger.info(f"==== 降维到 k={k} ====")
    # 2.1 UMAP 嵌入
    reducer = umap.UMAP(n_components=k, random_state=42)
    Z_train = reducer.fit_transform(X_emb_train)
    Z_test  = reducer.transform(X_emb_test)

    # 2.2 构建“解码+回归”联合模型
    inp = layers.Input(shape=(k,), name='Z_input')
    # 解码器：Z -> X_hat
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(128, activation='relu')(x)
    X_hat = layers.Dense(X.shape[1], name='reconstruction')(x)
    # 回归头：Z -> Y_pred
    y = layers.Dense(32, activation='relu')(inp)
    Y_pred = layers.Dense(1, name='prediction')(y)

    model = Model(inp, [X_hat, Y_pred], name=f"umap_decoder_k{k}")
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss={'reconstruction': losses.MeanSquaredError(),
              'prediction':   losses.MeanSquaredError()},
        loss_weights={'reconstruction': 1.0, 'prediction': 0.5}
    )
    # 回调
    es = callbacks.EarlyStopping('val_prediction_loss', patience=5, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau('val_prediction_loss', factor=0.5, patience=3)

    # 2.3 训练（使用降维集）
    history = model.fit(
        Z_train, {'reconstruction': X_emb_train, 'prediction': Y_emb_train},
        validation_data=(Z_test,  {'reconstruction': X_emb_test,  'prediction': Y_emb_test}),
        epochs=100, batch_size=256, callbacks=[es, rlrop], verbose=0
    )
    best_ep = np.argmin(history.history['val_prediction_loss']) + 1
    logger.info(f"  最佳 epoch: {best_ep}")

    # 2.4 评估
    X_test_rec, Y_test_pred = model.predict(Z_test, batch_size=256)
    mse_rec = mean_squared_error(X_emb_test, X_test_rec)
    mse_pred = mean_squared_error(Y_emb_test, Y_test_pred)
    r2      = r2_score(Y_emb_test, Y_test_pred)
    logger.info(f"  重构 MSE={mse_rec:.4f}, 预测 MSE={mse_pred:.4f}, R²={r2:.4f}")

    results.append({
        'k': k,
        'best_epoch': best_ep,
        'mse_rec': mse_rec,
        'mse_pred': mse_pred,
        'r2': r2
    })

# 3. 汇总 & 可视化
df = pd.DataFrame(results)
logger.info("所有结果:\n" + df.to_string(index=False))

# 绘图
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.lineplot(data=df, x='k', y='mse_rec', marker='o')
plt.title('重构 MSE vs k')
plt.xlabel('k')

plt.subplot(1,2,2)
sns.lineplot(data=df, x='k', y='r2', marker='s')
plt.title('预测 R² vs k')
plt.xlabel('k')
plt.ylim(0,1)  # R² 回归到 [0,1] 范围
plt.tight_layout()
plt.show()
