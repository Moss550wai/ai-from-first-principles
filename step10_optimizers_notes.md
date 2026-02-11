下面是完整的 **Step10 GitHub 笔记**（按统一模板整理）。保存为
`step10_optimizers_notes.md`

---

# Step10 — Momentum & Adam Optimizer

## Concept

普通梯度下降在复杂 loss 地形上存在三个核心问题：

* 在陡峭方向震荡
* 在平坦区域移动极慢
* 容易过早停在局部最优

因此需要改进优化算法，使训练 **更快、更稳定**。

本节核心：

* Momentum（动量）
* Adam（自适应优化器）

---

## Problem: Why Gradient Descent Is Inefficient

普通梯度下降更新规则：

```
w = w − lr * grad
```

特点：

* 每一步只依赖当前梯度
* 没有历史记忆
* 容易左右震荡
* 在平坦区域几乎停滞

直观理解：
像一个没有惯性的行人下山，每一步重新决定方向。

---

## Momentum Optimizer

Momentum 引入 **速度（velocity）** 概念。

核心思想：

过去的梯度应该影响现在的更新方向。

更新公式：

```
v = βv + (1−β)grad
w = w − lr * v
```

参数解释：

| 符号 | 含义           |
| -- | ------------ |
| v  | 梯度的指数滑动平均    |
| β  | 动量系数（通常 0.9） |

### Effect

Momentum 带来的改变：

* 平滑震荡
* 加速下降
* 更容易穿越小局部最优

直觉：
像滚下山的球，具有惯性。

---

## Adam Optimizer

Adam = Momentum + 自适应学习率。

问题来源：

不同参数的梯度规模不同：

* 梯度大 → 步长应更小
* 梯度小 → 步长应更大

Adam 同时记录两种信息：

```
m = 梯度平均（1st moment）
v = 梯度平方平均（2nd moment）
```

更新公式（简化版）：

```
m = β1 m + (1−β1) g
v = β2 v + (1−β2) g²

m̂ = m / (1 − β1^t)
v̂ = v / (1 − β2^t)

w = w − lr * m̂ / (sqrt(v̂) + ε)
```

默认参数：

```
β1 = 0.9
β2 = 0.999
ε = 1e−8
```

### Effect

Adam 的优势：

* 自动调整学习率
* 收敛快
* 对初始学习率不敏感
* 深度学习默认优化器

---

## Experiment Observation

在实验函数：

```
f(w) = w² + 3 sin(5w)
```

观察到：

* GD 收敛慢，但探索更充分
* Momentum 收敛更快
* Adam 收敛最快
* 三者可能停在不同局部最优

---

## Exploration vs Exploitation

实验揭示核心权衡：

| 优化器      | 特点        |
| -------- | --------- |
| GD       | 探索能力强，收敛慢 |
| Momentum | 平衡        |
| Adam     | 收敛快，早期承诺  |

这称为：

```
Exploration vs Exploitation Tradeoff
```

---

## Important Insight

在非凸优化中：

```
无法保证找到全局最优
```

深度学习的真实目标：

```
找到“足够好的最优解”
```

原因：

在高维空间中：

* 大多数局部最优性能相近
* 关键在于泛化能力，而不是最低 loss

---

## Key Takeaways

* Momentum 为梯度下降加入惯性
* Adam 为每个参数自适应学习率
* 快速收敛可能导致早期停留在局部最优
* 深度学习优化的目标是 good enough minima

