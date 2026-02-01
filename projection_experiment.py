import numpy as np
import matplotlib.pyplot as plt

# 1. 随机生成 100 个二维向量
np.random.seed(42)
points = np.random.randn(100, 2)

# 2. 随机选一个方向向量 v，并单位化
v = np.random.randn(2)
v = v / np.linalg.norm(v)

# 3. 投影：标量投影值（一维）
projections = points @ v   # shape: (100,)

# 4. 将一维投影“放回”到二维空间，方便可视化
projected_points = np.outer(projections, v)

# 5. 可视化
plt.figure(figsize=(10, 5))

# 原始空间
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], alpha=0.7)
plt.quiver(
    0, 0, v[0], v[1],
    angles='xy', scale_units='xy', scale=1,
    color='red', width=0.01
)
plt.title("Original 2D Points")
plt.axis('equal')
plt.grid(True)

# 投影后的世界
plt.subplot(1, 2, 2)
plt.scatter(projected_points[:, 0], projected_points[:, 1], alpha=0.7)
plt.plot(
    [-v[0]*5, v[0]*5],
    [-v[1]*5, v[1]*5],
    color='red'
)
plt.title("Projected onto direction v")
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()
