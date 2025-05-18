import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time
import matplotlib.font_manager as fm

# 查找系统中可用的中文字体
def find_chinese_fonts():
    chinese_fonts = []
    for font in fm.findSystemFonts():
        try:
            font_name = fm.FontProperties(fname=font).get_name()
            # 检查字体是否支持中文
            if 'Heiti' in font_name or 'Sim' in font_name or 'Micro Hei' in font_name:
                chinese_fonts.append(font_name)
        except:
            continue
    return chinese_fonts

# 获取可用的中文字体列表
chinese_fonts = find_chinese_fonts()
print("可用的中文字体:", chinese_fonts)

# 如果找到中文字体，则使用第一个可用的中文字体
if chinese_fonts:
    plt.rcParams["font.family"] = chinese_fonts[0]
else:
    print("警告: 未找到中文字体，图表中的中文可能无法正确显示")
    # 使用默认字体，但尝试解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""加载Excel数据并返回numpy数组"""


def load_data(file_path):
    """加载Excel数据并返回numpy数组"""
    try:
        df = pd.read_excel(file_path, header=None)
        return df.values
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


"""数据预处理：缺失值填充、异常值处理和标准化/归一化"""


def preprocess_data(data, iqr_threshold=1.5, normalize=True):
    """数据预处理：缺失值填充、异常值处理和标准化/归一化"""
    # 保存原始数据的副本
    original_data = data.copy()

    # 缺失值填充（使用前向填充）
    mask = np.isnan(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                if j > 0:
                    data[i, j] = data[i, j - 1]  # 前向填充
                else:
                    data[i, j] = 0  # 如果是第一列，填充0

    # 四分位距法异常值处理
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_threshold * IQR
    upper_bound = Q3 + iqr_threshold * IQR

    # 裁剪异常值
    data = np.clip(data, lower_bound, upper_bound)

    return data, original_data


"""执行SVD矩阵分解并返回重构矩阵"""


def matrix_decomposition(matrix, n_components):
    """执行SVD矩阵分解并返回重构矩阵"""
    svd = TruncatedSVD(n_components=n_components)
    transformed = svd.fit_transform(matrix)
    reconstructed = svd.inverse_transform(transformed)
    return reconstructed, svd


"""计算均方误差"""


def calculate_mse(original, reconstructed):
    """计算均方误差"""
    # 确保两个数组形状相同
    if original.shape != reconstructed.shape:
        min_shape = np.minimum(original.shape, reconstructed.shape)
        original = original[:min_shape[0], :min_shape[1]]
        reconstructed = reconstructed[:min_shape[0], :min_shape[1]]
    return np.mean((original - reconstructed) ** 2)


"""计算平均绝对误差"""


def calculate_mae(original, reconstructed):
    """计算平均绝对误差"""
    # 确保两个数组形状相同
    if original.shape != reconstructed.shape:
        min_shape = np.minimum(original.shape, reconstructed.shape)
        original = original[:min_shape[0], :min_shape[1]]
        reconstructed = reconstructed[:min_shape[0], :min_shape[1]]
    return np.mean(np.abs(original - reconstructed))


"""计算均方根误差"""


def calculate_rmse(original, reconstructed):
    """计算均方根误差"""
    # 确保两个数组形状相同
    if original.shape != reconstructed.shape:
        min_shape = np.minimum(original.shape, reconstructed.shape)
        original = original[:min_shape[0], :min_shape[1]]
        reconstructed = reconstructed[:min_shape[0], :min_shape[1]]
    return np.sqrt(np.mean((original - reconstructed) ** 2))


"""计算相对误差"""


def calculate_relative_error(original, reconstructed):
    """计算相对误差"""
    # 确保两个数组形状相同
    if original.shape != reconstructed.shape:
        min_shape = np.minimum(original.shape, reconstructed.shape)
        original = original[:min_shape[0], :min_shape[1]]
        reconstructed = reconstructed[:min_shape[0], :min_shape[1]]
    return np.mean(np.abs(original - reconstructed) / (np.abs(original) + 1e-10))


"""计算解释方差比例"""


def calculate_explained_variance(original, reconstructed):
    """计算解释方差比例"""
    # 确保两个数组形状相同
    if original.shape != reconstructed.shape:
        min_shape = np.minimum(original.shape, reconstructed.shape)
        original = original[:min_shape[0], :min_shape[1]]
        reconstructed = reconstructed[:min_shape[0], :min_shape[1]]
    return 1 - (np.var(original - reconstructed) / np.var(original))


"""计算压缩比"""


def calculate_compression_ratio(original, decomposed_elements):
    """计算压缩比"""
    # 首先计算原始数据大小
    original_size = np.prod(original.shape)

    if isinstance(decomposed_elements, tuple):  # 张量分解
        core, factors = decomposed_elements
        compressed_size = core.size
        for factor in factors:
            compressed_size += factor.size
    else:  # 矩阵分解或PCA
        model = decomposed_elements
        if hasattr(model, 'components_'):
            compressed_size = model.components_.size
            if hasattr(model, 'singular_values_'):
                compressed_size += model.singular_values_.size
            if hasattr(model, 'mean_'):
                compressed_size += model.mean_.size
        else:
            print(f"警告: 无法计算未知类型的压缩比: {type(model)}")
            return 1.0  # 默认返回1.0

    return original_size / compressed_size


"""计算存储空间节省率"""


def calculate_storage_saving_rate(original, decomposed_elements):
    """计算存储空间节省率"""
    original_size = np.prod(original.shape)
    if isinstance(decomposed_elements, tuple):  # 张量分解
        core, factors = decomposed_elements
        compressed_size = core.size

        # 调整因子矩阵的权重
        factor_weight = 0.5  # 可调整参数
        for factor in factors:
            compressed_size += int(factor.size * factor_weight)
    else:  # 矩阵分解或PCA
        model = decomposed_elements
        if hasattr(model, 'components_'):
            compressed_size = model.components_.size
            if hasattr(model, 'singular_values_'):
                compressed_size += model.singular_values_.size
            if hasattr(model, 'mean_'):
                compressed_size += model.mean_.size
        else:
            compressed_size = original_size  # 无法计算，默认不节省

    return (1 - compressed_size / original_size) * 100


"""评估降维后数据的聚类质量"""


def evaluate_clustering_quality(transformed_data):
    """评估降维后数据的聚类质量"""
    if transformed_data.shape[1] < 2:  # 至少需要两维数据
        return 0

    try:
        # 计算轮廓系数，评估聚类质量
        from sklearn.metrics import silhouette_score
        # 由于没有真实标签，我们假设所有样本属于同一类
        score = silhouette_score(transformed_data, np.zeros(transformed_data.shape[0]))
        return score
    except:
        return 0


"""粒子群算法中的粒子"""


class Particle:
    """粒子群算法中的粒子"""

    def __init__(self, dim, min_rank, max_rank):
        self.position = np.array([
            random.randint(min_rank[0], max_rank[0])
        ], dtype=np.float64)

        self.velocity = np.array([
            random.uniform(-10, 10)
        ])

        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        self.fitness_history = []


"""粒子群优化算法寻找最优组件数，平衡MSE和压缩比"""


def particle_swarm_optimization(train_matrix, test_matrix, original_train_matrix, original_test_matrix, max_iter=50,
                                num_particles=20, w=0.7, c1=1.4, c2=1.4, max_attempts=100):
    """粒子群优化算法寻找最优组件数，平衡MSE和压缩比"""
    dims = train_matrix.shape

    # 设置搜索范围
    min_rank = [1]
    max_rank = [min(500, min(dims))]  # 限制最大组件数

    print(f"矩阵维度: {dims}")
    print(f"搜索组件数范围: {max_rank[0]}")

    particles = [Particle(1, min_rank, max_rank) for _ in range(num_particles)]
    global_best_position = None
    global_best_cr = 0
    global_best_mse = float('inf')
    global_best_test_mse = float('inf')
    global_best_original_mse = float('inf')  # 新增：原始数据MSE
    global_best_original_test_mse = float('inf')  # 新增：原始测试数据MSE
    global_svd = None

    # 早停计数器
    no_improvement_count = 0
    best_original_mse = float('inf')
    best_original_test_mse = float('inf')

    # 尝试次数计数器
    total_attempts = 0

    print("开始粒子群优化搜索最优组件数...")

    for iteration in range(max_iter):
        start_time = time.time()
        iteration_improved = False

        for particle in particles:
            n_components = int(np.round(particle.position[0]))
            n_components = np.clip(n_components, min_rank[0], max_rank[0])

            total_attempts += 1

            try:
                # 对训练集进行矩阵分解和MSE计算
                train_reconstructed, svd = matrix_decomposition(train_matrix, n_components)
                train_mse = calculate_mse(train_matrix, train_reconstructed)

                # 对测试集进行矩阵分解和MSE计算
                test_reconstructed, _ = matrix_decomposition(test_matrix, n_components)
                test_mse = calculate_mse(test_matrix, test_reconstructed)

                cr = calculate_compression_ratio(train_matrix, svd)

                # 计算原始数据的重构和MSE
                original_train_reconstructed = svd.inverse_transform(svd.transform(original_train_matrix))
                original_mse = calculate_mse(original_train_matrix, original_train_reconstructed)

                original_test_reconstructed = svd.inverse_transform(svd.transform(original_test_matrix))
                original_test_mse = calculate_mse(original_test_matrix, original_test_reconstructed)

                # 只考虑原始训练集和测试集MSE都小于0.005的解
                if original_mse < 0.005 and original_test_mse < 0.005:
                    # 检查MSE是否过低，可能表示过拟合
                    if original_mse < 1e-10 or original_test_mse < 1e-10:
                        print(
                            f"警告: 检测到极低MSE (原始训练集: {original_mse:.10f}, 原始测试集: {original_test_mse:.10f})，可能存在过拟合")
                        # 计算相对误差作为额外验证
                        original_train_relative_error = calculate_relative_error(original_train_matrix,
                                                                                 original_train_reconstructed)
                        original_test_relative_error = calculate_relative_error(original_test_matrix,
                                                                                original_test_reconstructed)
                        print(
                            f"原始训练集相对误差: {original_train_relative_error:.10f}, 原始测试集相对误差: {original_test_relative_error:.10f}")

                        # 如果相对误差也很小，可能是合理的解
                        if original_train_relative_error < 1e-6 and original_test_relative_error < 1e-6:
                            print("相对误差也很低，解可能是合理的")
                        else:
                            print("相对误差较高，可能存在数值问题，忽略此解")
                            continue

                    if cr > particle.best_fitness:
                        particle.best_fitness = cr
                        particle.best_position = particle.position.copy()

                    if cr > global_best_cr:
                        global_best_cr = cr
                        global_best_position = n_components
                        global_best_mse = train_mse
                        global_best_test_mse = test_mse
                        global_best_original_mse = original_mse
                        global_best_original_test_mse = original_test_mse
                        global_svd = svd
                        print(
                            f"迭代 {iteration + 1}/{max_iter}: 新全局最优 - Components={global_best_position}, 预处理训练集MSE={global_best_mse:.6f}, 预处理测试集MSE={global_best_test_mse:.6f}, 原始训练集MSE={global_best_original_mse:.6f}, 原始测试集MSE={global_best_original_test_mse:.6f}, 压缩比={global_best_cr:.2f}x")
                        iteration_improved = True
                else:
                    print(
                        f"组件数 {n_components}: 原始训练集MSE={original_mse:.6f}, 原始测试集MSE={original_test_mse:.6f}，未达到要求（<0.005）")

            except Exception as e:
                print(f"计算组件数 {n_components} 时出错: {e}")

        # 更新粒子速度和位置
        for particle in particles:
            # 修正：分别生成两个随机数
            r1 = random.random()
            r2 = random.random()

            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (np.array([global_best_position]) - particle.position))

            # 限制最大速度，防止粒子跳出搜索空间
            particle.velocity = np.clip(particle.velocity, -50, 50)

            particle.position += particle.velocity
            particle.position = np.clip(particle.position, min_rank, max_rank)

        iteration_time = time.time() - start_time

        # 更新早停计数器
        if iteration_improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 如果找到了符合条件的解，且连续10次迭代没有改进，就提前结束
        if global_best_position is not None and no_improvement_count >= 10:
            print(f"早停: 连续{no_improvement_count}次迭代没有改进")
            break

        # 如果尝试次数太多，也提前结束
        if total_attempts >= max_attempts:
            print(f"达到最大尝试次数({max_attempts})，提前结束优化")
            break

        print(
            f"迭代 {iteration + 1}/{max_iter} 完成，耗时: {iteration_time:.2f}秒, 当前最优 - 预处理训练集MSE: {global_best_mse:.6f}, 预处理测试集MSE: {global_best_test_mse:.6f}, 原始训练集MSE: {global_best_original_mse:.6f}, 原始测试集MSE: {global_best_original_test_mse:.6f}, 压缩比: {global_best_cr:.2f}x")

    # 计算最终结果
    if global_best_position is not None:
        print(
            f"最终结果 - Components={global_best_position}, 预处理训练集MSE={global_best_mse:.6f}, 预处理测试集MSE={global_best_test_mse:.6f}, 原始训练集MSE={global_best_original_mse:.6f}, 原始测试集MSE={global_best_original_test_mse:.6f}, 压缩比={global_best_cr:.2f}x")
    else:
        print("警告: 未能找到满足原始训练集和测试集MSE均小于0.005的解")
        # 如果没有找到符合条件的解，返回最佳的可用解
        best_components = max_rank[0]  # 默认使用最大组件数
        print(f"使用最大组件数 {best_components} 作为备选方案")
        train_reconstructed, svd = matrix_decomposition(train_matrix, best_components)
        train_mse = calculate_mse(train_matrix, train_reconstructed)
        test_reconstructed, _ = matrix_decomposition(test_matrix, best_components)
        test_mse = calculate_mse(test_matrix, test_reconstructed)
        cr = calculate_compression_ratio(train_matrix, svd)
        original_train_reconstructed = svd.inverse_transform(svd.transform(original_train_matrix))
        original_mse = calculate_mse(original_train_matrix, original_train_reconstructed)
        original_test_reconstructed = svd.inverse_transform(svd.transform(original_test_matrix))
        original_test_mse = calculate_mse(original_test_matrix, original_test_reconstructed)

        global_best_position = best_components
        global_best_mse = train_mse
        global_best_test_mse = test_mse
        global_best_original_mse = original_mse
        global_best_original_test_mse = original_test_mse
        global_best_cr = cr
        global_svd = svd

        print(
            f"备选方案结果 - Components={global_best_position}, 预处理训练集MSE={global_best_mse:.6f}, 预处理测试集MSE={global_best_test_mse:.6f}, 原始训练集MSE={global_best_original_mse:.6f}, 原始测试集MSE={global_best_original_test_mse:.6f}, 压缩比={global_best_cr:.2f}x")

    return global_best_position, global_best_mse, global_best_test_mse, global_best_cr, global_best_original_mse, global_best_original_test_mse, global_svd


"""可视化矩阵分解结果"""


def visualize_results(
        original_matrix, matrix_recon, matrix_components, matrix_mse, matrix_cr, matrix_time,
        test_matrix, test_matrix_recon, test_matrix_mse,
        original_train_matrix, original_train_recon, original_mse,
        original_test_matrix, original_test_recon, original_test_mse,
        train_matrix_transformed=None, svd=None
):
    """可视化矩阵分解结果"""
    plt.figure(figsize=(15, 15))

    # 预处理训练数据矩阵分解结果
    plt.subplot(3, 2, 1)
    plt.imshow(matrix_recon, cmap='viridis')
    plt.colorbar()
    plt.title(
        f'SVD矩阵分解重构(预处理训练数据)\n组件数: {matrix_components}, MSE: {matrix_mse:.6f}, 压缩比: {matrix_cr:.2f}x')

    # 预处理训练数据矩阵分解误差
    plt.subplot(3, 2, 2)
    matrix_error = original_matrix - matrix_recon
    plt.imshow(matrix_error, cmap='coolwarm')
    plt.colorbar()
    plt.title('SVD分解误差矩阵(预处理训练数据)')

    # 预处理测试数据矩阵分解结果
    plt.subplot(3, 2, 3)
    plt.imshow(test_matrix_recon, cmap='viridis')
    plt.colorbar()
    plt.title(f'SVD矩阵分解重构(预处理测试数据)\nMSE: {test_matrix_mse:.6f}')

    # 预处理测试数据矩阵分解误差
    plt.subplot(3, 2, 4)
    test_matrix_error = test_matrix - test_matrix_recon
    plt.imshow(test_matrix_error, cmap='coolwarm')
    plt.colorbar()
    plt.title('SVD分解误差矩阵(预处理测试数据)')

    # 原始训练数据矩阵分解结果
    plt.subplot(3, 2, 5)
    plt.imshow(original_train_recon, cmap='viridis')
    plt.colorbar()
    plt.title(f'SVD矩阵分解重构(原始训练数据)\nMSE: {original_mse:.6f}')

    # 原始测试数据矩阵分解结果
    plt.subplot(3, 2, 6)
    plt.imshow(original_test_recon, cmap='viridis')
    plt.colorbar()
    plt.title(f'SVD矩阵分解重构(原始测试数据)\nMSE: {original_test_mse:.6f}')

    plt.tight_layout()
    plt.savefig('matrix_decomposition_result.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 性能对比表
    print("\n========== 矩阵分解性能指标 ==========")
    print(
        f"{'数据类型':<15} | {'训练集MSE':<10} | {'测试集MSE':<10} | {'压缩比':<10} | {'存储空间节省率(%)':<15} | {'计算时间(秒)'}")
    print("-" * 80)
    print(
        f"{'预处理数据':<15} | {matrix_mse:<10.6f} | {test_matrix_mse:<10.6f} | {matrix_cr:<10.2f} | {calculate_storage_saving_rate(original_matrix, svd):<15.2f} | {matrix_time:.2f}")
    print(
        f"{'原始数据':<15} | {original_mse:<10.6f} | {original_test_mse:<10.6f} | {matrix_cr:<10.2f} | {calculate_storage_saving_rate(original_train_matrix, svd):<15.2f} | {matrix_time:.2f}")
    print("-" * 80)

    # 计算其他评估指标
    matrix_rmse = calculate_rmse(original_matrix, matrix_recon)
    matrix_mae = calculate_mae(original_matrix, matrix_recon)
    matrix_rel_error = calculate_relative_error(original_matrix, matrix_recon)
    matrix_exp_var = calculate_explained_variance(original_matrix, matrix_recon)
    matrix_cluster_score = evaluate_clustering_quality(train_matrix_transformed)

    # 详细性能对比表
    print("\n========== 矩阵分解详细性能指标 ==========")
    print(
        f"{'数据类型':<15} | {'训练集MSE':<10} | {'测试集MSE':<10} | {'压缩比':<10} | {'存储空间节省率(%)':<15} | {'计算时间(秒)':<12} | {'解释方差(%)':<12} | {'相对误差':<10} | {'聚类质量'}")
    print("-" * 120)
    print(
        f"{'预处理数据':<15} | {matrix_mse:<10.6f} | {test_matrix_mse:<10.6f} | {matrix_cr:<10.2f} | {calculate_storage_saving_rate(original_matrix, svd):<15.2f} | {matrix_time:<12.2f} | {matrix_exp_var * 100:<12.2f} | {matrix_rel_error:<10.6f} | {matrix_cluster_score:.4f}")
    print(
        f"{'原始数据':<15} | {original_mse:<10.6f} | {original_test_mse:<10.6f} | {matrix_cr:<10.2f} | {calculate_storage_saving_rate(original_train_matrix, svd):<15.2f} | {matrix_time:<12.2f} | {calculate_explained_variance(original_train_matrix, original_train_recon) * 100:<12.2f} | {calculate_relative_error(original_train_matrix, original_train_recon):<10.6f} | {matrix_cluster_score:.4f}")
    print("-" * 120)

    # 可视化低维表示
    if train_matrix_transformed is not None:
        plt.figure(figsize=(6, 4))
        plt.scatter(train_matrix_transformed[:, 0], train_matrix_transformed[:, 1], alpha=0.5)
        plt.title('SVD低维表示')
        plt.xlabel('维度1')
        plt.ylabel('维度2')
        plt.tight_layout()
        plt.savefig('matrix_low_dim_representation.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # 1. 加载数据
    print("正在加载数据...")
    file_path = r"D:\新建文件夹 (2)\Data(1).xlsx"
    data = load_data(file_path)
    if data is None:
        print("数据加载失败，程序退出")
        return
    print(f"数据形状: {data.shape}")

    # 2. 数据预处理
    print("正在预处理数据...")
    processed_data, original_data = preprocess_data(data, iqr_threshold=1.5)

    # 3. 划分训练集和测试集
    print("正在划分训练集和测试集...")
    train_idx, test_idx = train_test_split(range(data.shape[0]), test_size=0.2, random_state=42)
    train_data = processed_data[train_idx]
    test_data = processed_data[test_idx]
    original_train_data = original_data[train_idx]
    original_test_data = original_data[test_idx]

    # 4. 准备矩阵分解数据
    print("正在准备矩阵分解数据...")
    train_matrix = train_data  # 使用原始二维数据
    test_matrix = test_data

    # 5. 矩阵分解（SVD）
    print("\n======= 开始矩阵分解 (SVD) =======")
    start_time = time.time()

    # 使用粒子群优化寻找最优组件数，增加最大尝试次数
    matrix_components, matrix_mse, test_matrix_mse, matrix_cr, original_mse, original_test_mse, svd = particle_swarm_optimization(
        train_matrix, test_matrix, original_train_data, original_test_data,
        max_iter=100,  # 增加最大迭代次数
        num_particles=30,  # 增加粒子数量
        max_attempts=500  # 增加最大尝试次数
    )

    matrix_time = time.time() - start_time

    print(f"\n最优组件数: {matrix_components}")
    print(f"预处理训练集MSE: {matrix_mse:.6f}")
    print(f"预处理测试集MSE: {test_matrix_mse:.6f}")
    print(f"原始训练集MSE: {original_mse:.6f}")
    print(f"原始测试集MSE: {original_test_mse:.6f}")
    print(f"SVD分解压缩比: {matrix_cr:.2f}x")

    # 执行SVD分解，确保所有需要的变量都被正确初始化
    matrix_recon, _ = matrix_decomposition(train_matrix, matrix_components)
    test_matrix_recon, _ = matrix_decomposition(test_matrix, matrix_components)  # 修复未解析的引用
    original_train_recon = svd.inverse_transform(svd.transform(original_train_data))
    original_test_recon = svd.inverse_transform(svd.transform(original_test_data))

    # 计算SVD降维后的低维表示
    train_matrix_transformed = svd.transform(train_matrix)

    # 6. 可视化结果
    print("\n正在生成可视化结果...")
    visualize_results(
        train_matrix, matrix_recon, matrix_components, matrix_mse, matrix_cr, matrix_time,
        test_matrix, test_matrix_recon, test_matrix_mse,
        original_train_data, original_train_recon, original_mse,
        original_test_data, original_test_recon, original_test_mse,
        train_matrix_transformed, svd
    )

    # 7. 导出重构数据
    print("\n正在导出重构数据...")
    try:
        # 导出预处理数据的重构结果
        recon_df = pd.DataFrame(matrix_recon)
        recon_df.to_excel('preprocessed_reconstructed_data.xlsx', index=False, header=False)

        # 导出原始数据的重构结果
        original_recon_df = pd.DataFrame(original_train_recon)
        original_recon_df.to_excel('original_reconstructed_data.xlsx', index=False, header=False)

        # 导出完整数据集的重构结果
        full_recon = svd.inverse_transform(svd.transform(processed_data))
        full_recon_df = pd.DataFrame(full_recon)
        full_recon_df.to_excel('combined_preprocessed_recon.xlsx', index=False, header=False)

        full_original_recon = svd.inverse_transform(svd.transform(original_data))
        full_original_recon_df = pd.DataFrame(full_original_recon)
        full_original_recon_df.to_excel('combined_original_recon.xlsx', index=False, header=False)

        print("重构数据已成功导出为Excel文件!")
    except Exception as e:
        print(f"导出重构数据时出错: {e}")

    print("分析完成！结果已保存到 'matrix_decomposition_result.png' 和相关Excel文件")


if __name__ == "__main__":
    main()