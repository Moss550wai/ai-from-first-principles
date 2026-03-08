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

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

lr = 0.01
epochs = 1000000

# 训练网络
for epoch in range(epochs):

    z1 = X @ W1 + b1
    h = relu(z1)
    z2 = h @ W2 + b2
    y_hat = sigmoid(z2)

    loss_grad = y_hat - y

    dW2 = h.T @ loss_grad
    db2 = np.sum(loss_grad, axis=0, keepdims=True)

    dh = loss_grad @ W2.T
    dz1 = dh * relu_grad(z1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W1 -= lr*dW1
    b1 -= lr*db1
    W2 -= lr*dW2
    b2 -= lr*db2

# ===== 画决策边界 =====

# 创建网格
xx, yy = np.meshgrid(
    np.linspace(-1, 2, 200),
    np.linspace(-1, 2, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# 让网络预测整个平面
z1 = grid @ W1 + b1
h = relu(z1)
z2 = h @ W2 + b2
pred = sigmoid(z2)
pred = pred.reshape(xx.shape)

# 画图
plt.contourf(xx, yy, pred, levels=50, alpha=0.6)
plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=100)
plt.title("Neural Network Decision Boundary")
plt.show()
