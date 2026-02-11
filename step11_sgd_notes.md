是的，这一份是 **Step11 的完整版本**。下面给你一次性给全，保证可以直接放进仓库。

保存为：`step11_sgd_notes.md`

---

# Step11 — Stochastic Gradient Descent (SGD)

## Concept

在前几步中，我们一直使用的是：

```
Batch Gradient Descent
```

即：

```
每一步使用全部训练数据计算梯度
```

这种方法梯度 **精确、稳定、平滑**，但在深度学习中存在关键问题：

* 计算成本极高
* 容易陷入局部最优
* 泛化能力较差

因此现实中的神经网络几乎从不使用纯 Batch GD。

---

## From Batch GD → SGD

真实训练方式：

```
每次只用一小批数据计算梯度
```

这就是：

```
Stochastic Gradient Descent (SGD)
```

更新公式变为：

```
w = w − lr * ∇L(batch)
```

关键变化：

```
梯度变成近似值，而不是精确值
```

---

## Why Use Approximate Gradients?

直觉上：

近似梯度似乎更差。

但实验和理论发现：

```
带噪声的梯度 → 更好的优化效果
```

核心原因：

梯度中引入了 **随机噪声**。

---

## Geometry of Loss Landscape

神经网络的 Loss Surface 是：

* 非凸（Non-convex）
* 高维
* 拥有大量局部最优

重要发现：

| 类型           | 特征  | 泛化能力 |
| ------------ | --- | ---- |
| Sharp minima | 窄而深 | 差    |
| Flat minima  | 宽而浅 | 好    |

目标不是最低 loss，而是：

```
找到 Flat Minima
```

---

## Noise as a Feature

SGD 的随机性带来：

```
梯度 = 真梯度 + 噪声
```

这种噪声具有关键作用：

* 帮助跳出窄谷（局部最优）
* 更容易进入宽谷（泛化更好）

因此：

```
SGD 的噪声不是 bug，而是 feature
```

---

## Optimization vs Generalization

深度学习训练目标：

```
不是找到全局最小值
```

而是：

```
找到泛化能力最好的解
```

SGD 正是实现这一点的关键机制。

---

## Batch vs Mini-Batch vs SGD

| 方法             | 每次使用数据量 | 特点   |
| -------------- | ------- | ---- |
| Batch GD       | 全部数据    | 稳定但慢 |
| SGD            | 1 个样本   | 噪声最大 |
| Mini-Batch SGD | 小批量     | 现实标准 |

现代深度学习实际使用：

```
Mini-Batch SGD
```

---

## Experimental Observation

在实验函数：

```
f(w) = w² + 3 sin(5w)
```

观察到：

* GD：平滑下降，但容易卡住
* SGD：路径抖动，但更容易跳出局部最优

说明：

```
噪声帮助探索更好的解空间
```

---

## Key Insight

深度学习成功的一个核心原因：

```
SGD 的随机噪声具有隐式正则化作用
```

它提升了模型的泛化能力。

---

## Key Takeaways

* Batch GD 梯度精确但容易陷入局部最优
* SGD 使用随机小批数据引入噪声
* 噪声帮助模型跳出窄谷
* 深度学习目标是 Flat Minima，而非最低 loss
* 现实训练使用 Mini-Batch SGD

---
