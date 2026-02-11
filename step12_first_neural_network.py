import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# XOR 数据
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([[0],[1],[1],[0]])

# 网络结构
input_size = 2
hidden_size = 4
output_size = 1

# 初始化参数
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# 激活函数
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s*(1-s)

lr = 0.1
epochs = 10000
losses = []

for epoch in range(epochs):

    # forward
    z1 = X @ W1 + b1
    h = relu(z1)
    z2 = h @ W2 + b2
    y_hat = sigmoid(z2)

    # loss (binary cross entropy)
    loss = -np.mean(y*np.log(y_hat+1e-8) + (1-y)*np.log(1-y_hat+1e-8))
    losses.append(loss)

    # backward
    dL_dy = y_hat - y
    dL_dW2 = h.T @ dL_dy
    dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)

    dL_dh = dL_dy @ W2.T
    dL_dz1 = dL_dh * relu_grad(z1)

    dL_dW1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    # update
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

# 训练结果
print("Predictions:")
print(np.round(y_hat,3))

# 画 loss
plt.plot(losses)
plt.title("Training Loss")
plt.show()
