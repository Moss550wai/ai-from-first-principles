import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. 生成一维数据
x = np.linspace(-5, 5, 50)
y = 2 * x + 1 + np.random.randn(50)  # 真正的世界 + 噪声

# 2. 随机初始化权重
w = np.random.randn()
b = np.random.randn()
lr = 0.01

# 3. 训练
for _ in range(2000):
    y_pred = w * x + b
    error = y_pred - y

    # 梯度（你现在不用记公式）
    dw = np.mean(error * x)
    db = np.mean(error)

    w -= lr * dw
    b -= lr * db

# 4. 可视化
plt.scatter(x, y)
plt.plot(x, w * x + b, color="red")
plt.title("Linear Regression Fit")
plt.show()
