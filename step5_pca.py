import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. 生成二维随机数据
points = np.random.randn(100, 2)
# 给数据加上偏移，使 x,y 有相关性
points[:, 1] = points[:, 0] * 0.5 + points[:, 1] * 0.5

# 2. 数据中心化
mean = np.mean(points, axis=0)
centered_points = points - mean

# 3. 计算协方差矩阵
cov = np.cov(centered_points, rowvar=False)

# 4. 求特征值和特征向量
eig_vals, eig_vecs = np.linalg.eigh(cov)
# 按特征值降序排列
idx = np.argsort(eig_vals)[::-1]
eig_vecs = eig_vecs[:, idx]

# 5. 投影到第一主成分方向
pc1 = eig_vecs[:, 0]
projections = centered_points @ pc1
projected_points = np.outer(projections, pc1)

# 6. 可视化
plt.figure(figsize=(10,5))

# 原始数据
plt.subplot(1,2,1)
plt.scatter(points[:,0], points[:,1], alpha=0.7)
plt.title("Original Data")
plt.axis('equal')
plt.grid(True)

# 投影到第一主成分
plt.subplot(1,2,2)
plt.scatter(projected_points[:,0]+mean[0], projected_points[:,1]+mean[1], alpha=0.7)
plt.title("Projected onto 1st Principal Component")
plt.axis('equal')
plt.grid(True)

# 画出第一主成分方向
plt.quiver(mean[0], mean[1], pc1[0], pc1[1], angles='xy', scale_units='xy', scale=2, color='red', width=0.01)

plt.tight_layout()
plt.show()
