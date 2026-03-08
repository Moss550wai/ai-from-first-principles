Step15 — Feature Learning
Core Idea

神经网络学习的不是答案，而是：

数据的新表示方式

Hidden Layer Function

每个隐藏神经元：

z = w·x + b
h = ReLU(z)


作用：

划分空间

激活特定区域

Representation Learning

隐藏层输出构成：

新的特征空间


输出层在这个新空间里进行线性分类。

Key Insight

深度网络的能力来自：

逐层组合简单特征


这比一次性用大量神经元更高效。