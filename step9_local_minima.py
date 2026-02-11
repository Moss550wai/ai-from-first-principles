import numpy as np
import matplotlib.pyplot as plt

# 多局部极小值函数
def loss(w):
    return w**2 + 3*np.sin(5*w)

# 梯度（导数）
def grad(w):
    return 2*w + 15*np.cos(5*w)

lr = 0.01
steps = 200

# 不同初始点
starts = [-2.5, -1.5, 0.5, 2.0]

# 画函数曲线
w_space = np.linspace(-3, 3, 400)
plt.plot(w_space, loss(w_space), label="Loss landscape")

for start in starts:
    w = start
    trajectory = [w]

    for i in range(steps):
        w = w - lr * grad(w)
        trajectory.append(w)

    trajectory = np.array(trajectory)
    plt.plot(trajectory, loss(trajectory), 'o-', label=f"start={start}")

plt.legend()
plt.title("Gradient Descent gets stuck in Local Minima")
plt.xlabel("w")
plt.ylabel("Loss")
plt.grid()
plt.show()
