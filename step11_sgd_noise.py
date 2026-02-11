import numpy as np
import matplotlib.pyplot as plt

# 同一个复杂loss
def loss(w):
    return w**2 + 3*np.sin(5*w)

def grad(w):
    return 2*w + 15*np.cos(5*w)

lr = 0.05
steps = 200
start = 2.5

# -------- 普通GD --------
w_gd = start
gd_path = []

for i in range(steps):
    w_gd -= lr * grad(w_gd)
    gd_path.append(w_gd)

# -------- SGD (加入噪声) --------
w_sgd = start
sgd_path = []

for i in range(steps):
    noise = np.random.normal(0, 2)  # 模拟mini-batch噪声
    noisy_grad = grad(w_sgd) + noise
    w_sgd -= lr * noisy_grad
    sgd_path.append(w_sgd)

# -------- 可视化 --------
w_space = np.linspace(-3,3,400)
plt.plot(w_space, loss(w_space), label="Loss")

plt.plot(gd_path, loss(np.array(gd_path)), label="GD")
plt.plot(sgd_path, loss(np.array(sgd_path)), label="SGD")

plt.legend()
plt.title("GD vs SGD (Noise helps escape local minima)")
plt.grid()
plt.show()
