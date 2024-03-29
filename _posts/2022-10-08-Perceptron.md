---
layout: post
read_time: true
show_date: true
title: "perceptron -- 感知机"
date: 2022-10-08 18:30:56 +0800
img: posts/20221130/ml.jpg
tags: [统计学习方法, 机器学习, machine learning, artificial intelligence]
category: 统计学习方法
author: Gary
description: "感知机是二分类的线性分类模型，是较为简单的模型。由Rosenblatt与1950年提出，是神经网络和支持向量机的基础。"
mathjax: yes
---

## 感知机 -- perceptron

> 感知机是二分类的线性分类模型，输出为+1和-1。
>
> 感知机学习旨在求出将训练数据进行线性划分的分离超平面，分为原始形式和对偶形式。

**感知机模型** ：$w$叫做权值（weight）或权值向量（weight vector），b叫做偏置（bias）
$$
f(x)=sign(w\cdot x+b)\\
sign(x)=
\begin{cases}
+1,x\geq0\\
-1,x<1
\end{cases}
$$
**感知机学习策略**：假设训练数据是线性可分的，感知机的学习的目标是求得一个能够将训练集正实例点和负实例点正确分开的分离超平面。

**损失函数：**

损失函数的一个自然选择是错误分类点的总数。但是这样的损失函数不是参数$w,b$的连续可导函数，不以优化，因此感知机采用误分类点距离超平面$S$的总距离，不考虑$\frac{1}{||w||}$就得到感知机学习的损失函数：
$$
-\frac{1}{||w||}\sum_{x_i}y_i(w\cdot x_i+b)\\
L(w,b) = -\sum_{x_i}y_i(w\cdot x_i+b)
$$
$||w||$是$w$的$L_2$范数。显然损失函数$L(w,b)$是非负的，误分类点越少，距离越近，损失函数值就越小。

**感知机学习算法**

给定训练集$T = {(x_1,y_1),\cdots,(x_N,y_N)} $，求参数$w,b$，使其为一下损失函数极小化问题的解：
$$
\min_{w,b}L(w,b)=-\sum_{x_i}y_i(w\cdot x_i+b)
$$
首先任选一个超平面$w_0,b_0$，然后用梯度下降法不断地极小化目标函数。假设误分类点集合$M$是固定的，那么损失函数$L(w,b)$的梯度：
$$
\nabla_wL(w,b)=-\sum_{x_i\in M}y_ix_i\\
\nabla_bL(w,b)=-\sum_{x_i\in M}y_i
$$
随机选取一个误分类点$(x_i,y_i)$，对$w,b$进行更新：
$$
w\leftarrow w+\eta y_ix_i\\
b\leftarrow b+\eta y_i
$$
$\eta$是步长，在统计学习中又称为学习率（learning rate)

> 算法 -- 感知机学习算法的原始形式
>
> 输入：训练集$T = {(x_1,y_1),\cdots,(x_N,y_N)}$；学习率$\eta$;
>
> 输出：$w,b$；感知机模型$f(x)=sign(w\cdot x+b)$。
>
> 1. 选取初值$w_0,b_0$；
>
> 2. 在训练集中选取数据$(x_i,y_i)$；
>
> 3. 如果$y_i(w\cdot x+b)\leq0$，
>    $$
>    w\leftarrow w+\eta y_ix_i\\
>    b\leftarrow b+\eta y_i
>    $$
>
> 4. 转至2，直至训练集中没有误分类点。



假设对$w,b$更新n次之后，其关于$(x_i,y_i)$增量分别为$\alpha_jy_jx_j$和$\alpha_iy_i$，$\alpha_i=n_i\eta$，$n_i$是点$(x_i,y_i)$被误分类的次数，当$\eta =1$时，$\alpha_i$表示第$i$个实例点由于误分而进行更新的次数。实例点更新次数越多，意味着距离分离超平面越近，也越难正确分类，这样的实例对学习结果影响最大。

> 算法 -- 感知机学习算法的对偶形式
>
> 输入：线性可分的数据集$T$；学习率$\eta$；
>
> 输出：$\alpha,b$；感知机模型$f(x)=sign(\sum_{j=1}^{N}\alpha_jy_jx_j\cdot x+b)$，其中$\alpha=(\alpha_1,\cdots,\alpha_N)^T$。
>
> 1. $\alpha\leftarrow 0,b\leftarrow 0$；
>
> 2. 在训练集中选取数据$x_i,y_i$；
>
> 3. 如果$y_i(\sum_{j=1}^{N}\alpha_jy_jx_j\cdot x+b)\le0$，
>    $$
>    \alpha_i\leftarrow\alpha_i +\eta\\
>    b\leftarrow b+\eta y_i
>    $$
>    
>
> 4. 转至2直到没有误分类数据。

对偶形式中训练实例仅以内积形式存在，可以预先将训练集中实例间的内积计算，即Gram矩阵（Gram matrix):
$$
G=[x_i\cdot x_j]_{N\times N}
$$

**例题**

$训练数据集中正实例点x_1=(3,3)^T,x_2=(4,3)^T，负实例点x_3=(1,1)^T，求感知机模型$

 > 原始形式：
 > 构建最优化问题：
 > $$
 > \min_{w,b}L(w,b) = -\sum_{x_i\in M}y_i(w\cdot x_i+b)
 > $$
 > 取$w_0=0,b_0=0$
 >
 > 对$x_1=(3,3)^T$，$y_1(w_0\cdot x_1+b_0)=0$，未被正确分类，更新$w,b$
 > $$
 > w_1 = w_0+y_1x_1 = (3,3)^T,b_1=b_0+y_1=1
 > $$
 > 得到线性模型
 > $$
 > w_1\cdot x+b_1 = 3x^{(1)}+3x^{(2)}+1
 > $$
 > 对$x_1,x_2$，显然，$y_i(w_1\cdot x_i+b_1)>0$，正确分类，对$x_3$错误分类，更新$w,b$
 > $$
 > w_2 = w_1+y_3x_3 = (2,2)^T,b_2=b_1+y_3=0
 > $$
 > 得到线性模型
 > $$
 > w_1\cdot x+b_1 = 2x^{(1)}+2x^{(2)}
 > $$
 > 直到，对所有的数据点都正确分类，损失函数达到极小。
 >
 > 得到：$w_7=(1,1)^T,b_7=-3$
 >
 > 感知机模型为：$f(x)=sign(x^{(1)}+x^{(2)}-3)$
 >
 > *不同的初值或选取不同的误分类点，解可以不同*

> 对偶形式：
>
> 取$\alpha_i=0,i=1,2,3,b=0,\eta =1$
>
> 计算Gram矩阵
> $$
> G = \begin{bmatrix}
> 18&21&6\\
> 21&25&7\\
> 6&7&2
> \end{bmatrix}
> $$
> 对$x_1=(3,3)^T,y_1(\sum_{j=1}^{N}\alpha_jy_jx_j\cdot x_1+b_1)=0$被误分，更新$\alpha_1,b$
>
> $\alpha_1=\alpha_1+1=1,\alpha_2=0,\alpha_3=0,b=b+y_1=1$
>
> 得到模型：
> $$
> \sum_{j=1}^{3}\alpha_jy_jx_j\cdot x_i+b = x_1\cdot x_i+1
> $$
> 对$x_3=(1,1)^T,y_3(x_1\cdot x_3+1)=-1$被误分，更新$\alpha_3,b$
>
> $\alpha_1=1,\alpha_2=0,\alpha_3 = \alpha_3+1=1,b=b+y_3=0$
>
> 迭代，直到没有误分数据。

**python实现**

```python
import numpy as np
import matplotlib.pyplot as plt


def create_datasets():
    data = np.array([[3,3],[4,3],[1,1],[2,3],[4,5],[2,0]])
    label = np.array([1,1,-1,-1,1,-1])
    return data, label


class Perceptron():
    def __init__(self,samples, label, eta=1):
        self.samples = samples
        self.label = label
        self.eta = eta
        self.w = np.zeros(samples.shape[1])
        self.b = 0
        # for dual model
        self.alpha = np.zeros(samples.shape[0])
        self.G = np.dot(samples,samples.T)

    def train_origin(self):
        # 原始形式
        ix_all = np.array([0])
        while np.any(np.array(ix_all)<=0):
            ix_all = []        
            for i,sample in enumerate(self.samples):
                ix = self.label[i]*(np.dot(self.w.T,sample)+self.b)
                ix_all.append(ix)
                if ix<=0:
                    self.w += self.eta*self.label[i]*sample
                    self.b += self.eta*self.label[i]
        return self.w,self.b

    def train_dual(self):
        # 对偶形式
        end = False
        while not end:
            count = 0
            for i,sample in enumerate(self.samples):
                ix = self.label[i]*(np.sum(self.alpha.T*self.label*self.G[i])+self.b)
                if ix<=0:
                    count += 1
                    self.alpha[i] += self.eta
                    self.b += self.eta*self.label[i]
            if count==0:
                end = True
        self.w = np.sum(self.alpha*self.label*self.samples.T,axis=1)
        self.b = np.sum(self.alpha*self.label)      
        return self.w,self.b

    def show(self, model='origin'):
        plt.figure()
        plt.scatter(self.samples[:,0],self.samples[:,1],c=self.label)
        x = np.linspace(self.samples[:,0].min(),self.samples[:,0].max(),10)
        if model=='origin':
            self.w, self.b = self.train_origin()
        else:
            self.w, self.b = self.train_dual()
        y = (-self.b-self.w[0]*x)/self.w[1]
        plt.plot(x,y)
        plt.show()

if __name__=='__main__':
    data, label = create_datasets()
    myper = Perceptron(data, label)
    myper.show()

```
