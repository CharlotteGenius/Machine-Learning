#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:54:05 2019

@author: xiangyinyu
"""


"=======================!!!Don't Try to Run this Code!!!======================"


# NOTE:

# Simple Linear Regression:
"""
    y = b0 + b1*x1
    
    y:Dependent variable (DV)
    x1: Independent variable (IV)
    b1: Coefficient
    b0: Constant

For example, we know datasets of Salary vs. Experience,
Using this model, we could say: 
    Salary = b0 + b1 * Experience

That is to draw a line that fits into the 2 dimension chart.
while b0 is the cross point with the axis,
which meanas that, if experiance is 0, the salary would be b0.
b1 is the slope od this model,
meaning that, with 1 more year experience, I can get b1 more salary.
"""

# How to find a good linear model?
"""
Now we have a linear model, and for a certain point,
the observed value is yi, and the modeled value is yi` 
and we want SUM((yi-yi`)^2) to be the minimum.
"""
# -------------------------->> Ordinary Least Square


# Python sklearn library NOTE:
# https://zhuanlan.zhihu.com/p/42297868
"""注意这是数据预处理中的方法：

Fit(): Method calculates the parameters μ and σ and saves them as internal objects.
解释：简单来说，就是求得训练集X的均值，方差，最大值，最小值这些训练集X固有的属性。可以理解为一个训练过程

Transform(): Method using these calculated parameters apply the transformation to a particular dataset.
解释：在Fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）。

Fit_transform(): joins the fit() and transform() method for transformation of dataset.
解释：fit_transform是fit和transform的组合，既包括了训练又包含了转换。


transform()和fit_transform()二者的功能都是对数据进行某种统一处理
比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等

fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，
如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，
从而实现数据的标准化、归一化等等。

根据对之前部分trainData进行fit的整体指标，
对剩余的数据（testData）使用同样的均值、方差、最大最小值等指标进行转换transform(testData)，
从而保证train、test处理方式相同。所以，一般都是这么用:"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_tranform(X_train)
sc.tranform(X_test)
"""
Note:
必须先用fit_transform(trainData)，之后再transform(testData)
如果直接transform(testData)，程序会报错
如果fit_transfrom(trainData)后，使用fit_transform(testData)而不是transform(testData)，
虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。(一定要避免这种情况)
"""




# Multiple Linear Regression:



