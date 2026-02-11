import numpy as np

# 数据
X = np.array([
    [2,5,1],
    [3,6,2],
    [1,4,3],
    [4,7,1]
])

y = np.array([15,18,10,22])

# 随机权重
w = np.random.randn(3)

# 预测
y_pred = X @ w

# MSE
loss = np.mean((y_pred - y) ** 2)

print("预测:", y_pred)
print("Loss:", loss)
