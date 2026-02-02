import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 随机生成 100 个二维点
points = np.random.randn(100, 2)

# 定义非线性激活函数
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 选择激活函数（这里以 ReLU 为例）
transformed_points = relu(points)

# 随机选一个方向向量
v = np.random.randn(2)
v = v / np.linalg.norm(v)

# 投影
projections = transformed_points @ v
projected_points = np.outer(projections, v)

# 可视化
plt.figure(figsize=(10,5))

# 原始空间
plt.subplot(1,2,1)
plt.scatter(points[:,0], points[:,1], alpha=0.7)
plt.title("Original Points")
plt.grid(True)

# 非线性映射后的投影
plt.subplot(1,2,2)
plt.scatter(projected_points[:,0], projected_points[:,1], alpha=0.7)
plt.title("Projected after ReLU")
plt.grid(True)

plt.tight_layout()
plt.show()
