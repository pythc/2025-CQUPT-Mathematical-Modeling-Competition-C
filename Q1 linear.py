import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载与预处理 ====================
print("======= 数据加载与预处理 =======")
data_a = pd.read_excel("A.xlsx", header=None)
data_b = pd.read_excel("B.xlsx", header=None)

# 确保数据对齐
data_a = data_a.iloc[:len(data_b)]
scaler = StandardScaler()
data_a_scaled = scaler.fit_transform(data_a)

# ==================== 数据分割 ====================
X_train, X_test, y_train, y_test = train_test_split(
    data_a_scaled, data_b, test_size=0.2, random_state=42)
y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

# ==================== 基础线性回归 ====================
print("\n======= 基础线性回归训练 =======")
base_model = LinearRegression()
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)

# ==================== 评估指标计算 ====================
def evaluate_model(y_true, y_pred, model_name=""):
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Pearson_r': pearsonr(y_true, y_pred)[0]
    }
    print(f"\n{model_name}评估指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    return metrics

base_metrics = evaluate_model(y_test, y_pred, "基础线性回归")

# ==================== 改进建议实现 ====================
# 1. 残差分析
print("\n======= 残差分析 =======")
residuals = y_test - y_pred

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("残差 vs 预测值")
plt.xlabel("预测值")
plt.ylabel("残差")

plt.subplot(132)
sns.histplot(residuals, kde=True, bins=30)
plt.title("残差分布")

plt.subplot(133)
from statsmodels.graphics.gofplots import qqplot
qqplot(residuals, line='s', ax=plt.gca())
plt.title("Q-Q图")

plt.tight_layout()
plt.show()

# 2. 共线性检测 (VIF)
print("\n======= 共线性检测 (VIF) =======")
vif_data = pd.DataFrame()
vif_data["Variable"] = [f"X{i}" for i in range(1, data_a.shape[1]+1)]
vif_data["VIF"] = [variance_inflation_factor(data_a_scaled, i)
                  for i in range(data_a_scaled.shape[1])]
print("VIF > 10的变量（存在共线性风险）:")
print(vif_data[vif_data["VIF"] > 10].sort_values("VIF", ascending=False))

# 3. 异常值检测 (改进版 - 使用离群值检测代替Cook距离)
print("\n======= 异常值检测 =======")
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(np.c_[data_a_scaled, data_b.values.ravel()])
outlier_indices = np.where(outliers == -1)[0]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, c='b', label='正常值')
plt.scatter(outlier_indices[outlier_indices < len(y_test)],
           y_test[outlier_indices[outlier_indices < len(y_test)]],
           c='r', label='检测到的异常值')
plt.title("异常值检测结果 (Isolation Forest)")
plt.xlabel("样本索引")
plt.ylabel("目标值")
plt.legend()
plt.show()

print(f"检测到的异常值数量: {len(outlier_indices)}")

# 4. 特征选择 (Lasso)
print("\n======= Lasso特征选择 =======")
lasso = LassoCV(cv=5).fit(X_train, y_train)
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"原始特征数: {X_train.shape[1]}, 筛选后特征数: {len(selected_features)}")
print("重要特征索引:", selected_features)

# 5. 稳健回归 (Huber)
print("\n======= 稳健回归训练 =======")
robust_model = HuberRegressor().fit(X_train, y_train)
y_pred_robust = robust_model.predict(X_test)
robust_metrics = evaluate_model(y_test, y_pred_robust, "稳健回归")

# 6. 不确定性量化 (分位数回归)
print("\n======= 不确定性量化 =======")
quantile_model = GradientBoostingRegressor(
    loss='quantile', alpha=0.95, random_state=42)
quantile_model.fit(X_train, y_train)
upper_bound = quantile_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='预测值')
plt.scatter(y_test, upper_bound, alpha=0.3, c='r', label='95%分位数边界')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.legend()
plt.title("预测不确定性可视化")
plt.show()

# ==================== 模型对比 ====================
print("\n======= 模型性能对比 =======")
results = pd.DataFrame({
    'Metric': ['MSE', 'MAE', 'R2', 'Pearson_r'],
    'Base Model': [base_metrics['MSE'], base_metrics['MAE'],
                  base_metrics['R2'], base_metrics['Pearson_r']],
    'Robust Model': [robust_metrics['MSE'], robust_metrics['MAE'],
                    robust_metrics['R2'], robust_metrics['Pearson_r']]
})
print(results)

# ==================== 最终模型输出 ====================
final_model = base_model  # 根据对比结果选择最佳模型

print("\n======= 最终回归方程 =======")
equation = "Y = "
for i, coef in enumerate(final_model.coef_):
    equation += f"{coef:.6f}*X{i+1} + "
equation += f"{final_model.intercept_:.6f}"
print(equation)

# 交叉验证验证稳定性
cv_scores = cross_val_score(final_model, data_a_scaled, data_b.values.ravel(),
                           cv=5, scoring='r2')
print(f"\n交叉验证R²: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")