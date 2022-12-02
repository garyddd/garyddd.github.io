---
layout: post
read_time: true
show_date: true
title: "naïve Bayes -- 朴素贝叶斯"
date: 2022-10-16 18:30:56 +0800
img: posts/20221130/ml.jpg
tags: [统计学习方法, 机器学习, machine learning, artificial intelligence]
category: 统计学习方法
author: Gary
mathjax: yes
---

# 朴素贝叶斯

---

> 朴素贝叶斯（naïve Bayes）法是基于贝叶斯定理和特征条件独立假设的分类方法。
>
> 对于给定的训练数据集，首先基于特征条件独立假设学习输入输出的联合概率分布；然后基于此概率模型，对输入的x计算其后验概率最大的输出y。属于生成模型。

## 基本方法

训练数据集：$T = {(x_1,y_1),\cdots,(x_N,y_N)}$，是由$P(X,Y)$及独立同分布产生。

朴素贝叶斯法通过训练数据集学习联合分布概率$P(X,Y)$。首先学习先验概率分布和条件概率分布，先验概率分布：
$$
P(Y=c_k),k = 1,2,\cdots,K
$$
条件分布概率：
$$
P(X=x|Y=c_k) = P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=X^{(n)}|Y=c_k), k=1,2,\cdots,K
$$
由于朴素贝叶斯的强假设，即条件独立性假设。因此：
$$
P(X=x|Y=c_k) = \prod_{j=1}^{n}P(X^{(j)}=x^{(j)}|Y=c_k)
$$
分类是通过计算后验概率，即$P(Y=c_k|X=x)$，将后验概率最大的类作为x的类输出。后验概率计算：
$$
P(Y=c_k|X=x) = \frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_{k}P(X=x|Y=c_k)P(Y=c_k)}
$$
将先验概率和条件概率代入到后验概率公式可得，朴素贝叶斯分类器可表示为：
$$
y = f(x)=arg \max_{c_k}\frac{P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_kP(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}
$$
由于对所有的$c_k$，分母是相同的，因此：
$$
y = f(x)=arg \max_{c_k}P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)
$$

## 参数估计

### 极大似然估计

先验概率的极大似然估计：
$$
P(Y=c_k) = \frac{\sum_{i=1}^{N}I(y_i=c_k)}{N}, k=1,2,\cdots,K
$$
条件概率的极大似然估计：
$$
P(X^{(j)}=a_{jl}|Y=c_k) = \frac{\sum_{i=1}^{N}I(x_{i}^{(j)}=a_{jl},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}
$$

### 贝叶斯估计

极大似然估计可能会出现所要估计的概率值为0的情况，会影响到后验概率的计算结果。

条件概率的贝叶斯估计：
$$
P_\lambda(X^{(j)}=a_{jl}|Y=c_k) = \frac{\sum_{i=1}^{N}I(x_{i}^{(j)}=a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^{N}I(y_i=c_k)+S_j\lambda}
$$
引入的$\lambda\geq0$，等价于在随机变量哥哥取值的频数上赋予一个正数，用于平滑概率为0的条件。取0时，等价于极大似然估计；取1时，成为拉普拉斯平滑（Laplacian smoothing）。其满足概率分布要求，即$P>0$，概率和为1。

同样，先验分布的贝叶斯估计：
$$
P(Y=c_k) = \frac{\sum_{i=1}^{N}I(y_i=c_k)+\lambda}{N+K\lambda}, k=1,2,\cdots,K
$$


### 朴素贝叶斯算法

> 算法：朴素贝叶斯算法
>
> 输入：训练样本，实例x
>
> 输出：实例x的分类
>
> 1. 计算先验概率和条件概率：
>    $$
>    P(Y=c_k) = \frac{\sum_{i=1}^{N}I(y_i=c_k)}{N}, k=1,2,\cdots,K\\
>    P(X^{(j)}=a_{jl}|Y=c_k) = \frac{\sum_{i=1}^{N}I(x_{i}^{(j)}=a_{jl},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}
>    $$
>    
>
> 2. 对给定的实例，计算
>    $$
>    P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)
>    $$
>
> 3. 确定实例x的类
>    $$
>    y = arg \max_{c_k}P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)
>    $$

### python实现

```python
import numpy as np
import pandas as pd
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

def train_data():
    data = pd.DataFrame([[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],['S','M','M','S','S','S','M','M','L','L','L','M','M','L','L'],[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3][::-1]]).T
    return data

class naiveBayes:
    def __init__(self, data, label):
        self.train_data = data
        self.label = label
    
    def fit(self, test_data):
        # 极大似然估计
        p_prior = dict(pd.Series(label).value_counts())
        p_condi = {}
        for col in self.train_data.columns:
            df = self.train_data[[col]]
            df['label'] = self.label
            p_condi.update(dict(df.value_counts()))
        y = []
        for y_i in np.unique(self.label):
            y.append(p_prior[y_i]/self.train_data.shape[0]*reduce(lambda x0,x1:x0*x1,[p_condi[(i,y_i)]/p_prior[y_i] for i in test_data]))
        return np.unique(self.label)[y.index(max(y))]
    
    def fit1(self, test_data):
        # 贝叶斯估计
        
data = train_data()
label = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
mybayes = naiveBayes(data, label)
%time mybayes.fit([2,'S',1])
```

