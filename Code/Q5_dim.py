import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, Model, optimizers, losses, callbacks
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# 1. 数据加载
X = pd.read_excel('5-X.xlsx', header=None).values
Y = pd.read_excel('5-Y.xlsx', header=None).values.ravel()
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
logger.info(f"训练样本: {X_train.shape}, 验证样本: {X_val.shape}")

# 2. 模型构造
def build_autoencoder(input_dim, latent_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    z = layers.Dense(latent_dim, name='latent')(x)
    # 解码器
    x_dec = layers.Dense(64, activation='relu')(z)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Dropout(0.2)(x_dec)
    x_dec = layers.Dense(128, activation='relu')(x_dec)
    rec = layers.Dense(input_dim, name='reconstruction')(x_dec)
    # 预测分支
    y_pred = layers.Dense(32, activation='relu')(z)
    y_pred = layers.Dense(1, name='prediction')(y_pred)
    return Model(inp, [rec, y_pred], name=f'AE_k{latent_dim}')

# 3. 批量实验
latent_dims = [40, 60, 80, 100]
loss_weights = [(1.0, 0.5), (1.0, 1.0), (1.0, 2.0)]
results = []

for k in latent_dims:
    for alpha, beta in loss_weights:
        logger.info(f"实验: latent_dim={k}, α={alpha}, β={beta}")
        model = build_autoencoder(X.shape[1], k)
        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss={'reconstruction': losses.MeanSquaredError(),
                  'prediction':   losses.MeanSquaredError()},
            loss_weights={'reconstruction': alpha, 'prediction': beta}
        )
        es = callbacks.EarlyStopping('val_prediction_loss', patience=5, restore_best_weights=True)
        rlrop = callbacks.ReduceLROnPlateau('val_prediction_loss', factor=0.5, patience=3)
        hist = model.fit(
            X_train, {'reconstruction': X_train, 'prediction': y_train},
            validation_data=(X_val, {'reconstruction': X_val, 'prediction': y_val}),
            epochs=100, batch_size=256, callbacks=[es, rlrop], verbose=0
        )
        best_ep = np.argmin(hist.history['val_prediction_loss']) + 1
        X_val_rec, Y_val_pred = model.predict(X_val, batch_size=256)
        mse_rec = mean_squared_error(X_val, X_val_rec)
        mse_pred = mean_squared_error(y_val, Y_val_pred)
        r2 = r2_score(y_val, Y_val_pred)
        logger.info(f"→ best_epoch={best_ep}, rec_MSE={mse_rec:.4f}, pred_MSE={mse_pred:.4f}, R²={r2:.4f}")
        results.append({'k':k,'α':alpha,'β':beta,'best_epoch':best_ep,
                        'mse_rec':mse_rec,'mse_pred':mse_pred,'r2':r2})

# 4. 汇总 & 可视化
df = pd.DataFrame(results)
logger.info("所有结果:\n" + df.to_string(index=False))

# 修正 pivot 调用：使用命名参数
pivot = df.pivot(index='k', columns='β', values='r2')
plt.figure(figsize=(6,5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis')
plt.title('预测 R² 热力图')
plt.ylabel('latent_dim k')
plt.xlabel('β (prediction weight)')
plt.show()
