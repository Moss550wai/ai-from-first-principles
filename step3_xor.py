import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 1. 生成 XOR 数据
# 四个点，也可以用 100 个随机点
points = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 1, 1, 0])  # XOR 输出

# 2. 非线性映射，例如 x*y 或 x^2 + y^2
# 这里用简单的 ReLU-like映射示意
def relu(x):
    return np.maximum(0, x)

transformed_points = np.column_stack([
    relu(points[:, 0]),
    relu(points[:, 1]),
    relu(points[:, 0] * points[:, 1])  # 加入交互项
])

# 3. 随机选择一个方向向量投影
v = np.random.randn(transformed_points.shape[1])
v = v / np.linalg.norm(v)
projections = transformed_points @ v
projected_points = np.outer(projections, v)

# 4. 可视化
plt.figure(figsize=(10, 5))

# 原始 XOR
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c=labels, s=100, cmap='coolwarm')
plt.title("Original XOR Points")
plt.grid(True)

# 投影后的非线性空间
plt.subplot(1, 2, 2)
plt.scatter(projected_points[:, 0], projected_points[:, 1], c=labels, s=100, cmap='coolwarm')
plt.title("Projected after Non-linear Mapping")
plt.grid(True)

plt.tight_layout()
plt.show()
