import numpy as np
import matplotlib.pyplot as plt

# ====== 可中断训练 ======
try:
    # 1️⃣ 数据：XOR 4 个点
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])  # XOR 输出

    # 2️⃣ 初始化权重和偏置
    np.random.seed(42)
    w = np.random.randn(2)
    b = np.random.randn()
    lr = 0.1
    epochs = 1000

    # 3️⃣ 激活函数 ReLU
    def relu(z):
        return np.maximum(0, z)

    # 4️⃣ 前向 + 训练循环
    for epoch in range(epochs):
        # 前向传播
        z = x @ w + b  # shape (4,)
        y_pred = relu(z)

        # 误差
        error = y_pred - y

        # 梯度（简单版，非严格微分）
        dw = np.mean(error[:, np.newaxis] * x, axis=0)
        db = np.mean(error)

        # 参数更新
        w -= lr * dw
        b -= lr * db

        # 每 100 次打印一次
        if (epoch+1) % 100 == 0:
            loss = np.mean(error**2)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, w: {w}, b: {b}")

    # 5️⃣ 可视化
    plt.figure(figsize=(6,6))
    for i in range(4):
        color = 'red' if y[i]==1 else 'blue'
        plt.scatter(x[i,0], x[i,1], color=color, s=100)
    
    # 绘制决策边界
    xx = np.linspace(-0.5, 1.5, 100)
    yy = -(w[0]*xx + b)/w[1]
    plt.plot(xx, yy, color='green', label='Decision boundary (linear)')
    plt.title("XOR points & linear decision boundary")
    plt.grid(True)
    plt.legend()
    plt.show()

except KeyboardInterrupt:
    print("Training paused by user")
    print(f"Current parameters: w={w}, b={b}")
