import numpy as np
import matplotlib.pyplot as plt

# 复杂loss地形
def loss(w):
    return w**2 + 3*np.sin(5*w)

def grad(w):
    return 2*w + 15*np.cos(5*w)

lr = 0.05
steps = 150
start = 2.5

# -------- 普通GD --------
w_gd = start
gd_path = []

for i in range(steps):
    w_gd -= lr * grad(w_gd)
    gd_path.append(w_gd)

# -------- Momentum --------
w_m = start
v = 0
beta = 0.9
momentum_path = []

for i in range(steps):
    v = beta*v + (1-beta)*grad(w_m)
    w_m -= lr*v
    momentum_path.append(w_m)

# -------- Adam --------
w_a = start
m, v = 0, 0
beta1, beta2 = 0.9, 0.999
eps = 1e-8
adam_path = []

for t in range(1, steps+1):
    g = grad(w_a)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*(g**2)

    m_hat = m/(1-beta1**t)
    v_hat = v/(1-beta2**t)

    w_a -= lr*m_hat/(np.sqrt(v_hat)+eps)
    adam_path.append(w_a)

# -------- 可视化 --------
w_space = np.linspace(-3,3,400)
plt.plot(w_space, loss(w_space), label="Loss")

plt.plot(gd_path, loss(np.array(gd_path)), label="GD")
plt.plot(momentum_path, loss(np.array(momentum_path)), label="Momentum")
plt.plot(adam_path, loss(np.array(adam_path)), label="Adam")

plt.legend()
plt.title("Optimizer Comparison")
plt.grid()
plt.show()
