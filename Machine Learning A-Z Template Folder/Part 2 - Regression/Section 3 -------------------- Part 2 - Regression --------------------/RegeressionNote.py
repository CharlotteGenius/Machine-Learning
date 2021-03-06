#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:54:05 2019

@author: xiangyinyu
"""


"=======================!!!Don't Try to Run this Code!!!======================"


# NOTE:

'''--------------------------------Session 4--------------------------------'''

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
# -------------->> Ordinary Least Square


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



'''--------------------------------Session 5--------------------------------'''

# Multiple Linear Regression:
"""
   y = b0 + b1*x1 + b2*x2 + ... + bn*xn
   ^    ^       ^      ^             ^
   DV  Cons     IV     IV            IV
   
 There are multiple variables/coefficient that affect the result.

A Caveat:
    Assumptions of a Linear Regression:
        1. Linearity
        2. Homoscedasticity 方差齐性
        3. Multivariate normality 多元正态
        4. Independence of errors
        5. Lack of multicollinearity 缺乏多重性
"""


# Dummy Variables
"""
   State   ---->>   NY  CA
    NY              1   0
    CA              0   1
    CA              0   1
    NY              1   0
    CA              0   1
                   |-----|  These are Dummy Variables
                   
Transfer Catagory variables to numerical variables.
1 is like switch on and 0 is off.
** You can't include more then one dummy variable in a same model
    y = b0 + b1*x1 + b2*x2 + ... + b3*D1 + b4*D2
And: D1 + D2 = 1
"""

# the significance level
"""
the significance level is a measure of how certain we want to be about our results 

- low significance values correspond to a low probability that 
 the experimental results happened by chance, and vice versa.
 
- Significance levels are written as a decimal (such as 0.01),
 which corresponds to the percent chance that random sampling would produce a difference 
 as large as the one you observed if there was no underlying difference in the populations.

- By convention, scientists usually set the significance value 
 for their experiments at 0.05, or 5 percent.
 This means that experimental results that meet this significance level have, at most, 
 a 5% chance of being reproduced in a random sampling process. 
"""

# p-value
"""
P values are used to determine whether the results of their experiment 
 are within the normal range of values for the events being observed.
 Usually, if the P value of a data set is below a certain pre-determined amount (like, for instance, 0.05), 
 scientists will reject the "null hypothesis" of their experiment.
 In other words, they'll rule out the hypothesis that the variables of their experiment 
 had no meaningful effect on the results.
 Today, p values are usually found on a reference table by first calculating a chi square value.

Step 1. Determine expected results.
Step 2. Observed results.
Step 3. Degrees of freedom.  n-1    n: number of variables.
Step 4. Compare expected results to observed results with chi square.
        x2 = Σ((o-e)2/e)    o: observed value; e: expected value
Step 5. Choose a Significant Level (say 0.05)
Step 6. Use a chi square distribution table to approximate your p-value.
Step 7. Decide whether to reject or keep your null hypothesis
"""

# 5 methods of building models
"""
1. All-in
    - Prior knowledge;
    - You have to; (bank, company needs)
    - Prepare for backword eimination
    
2. Backward Elimination
    Step 1. Select a significant level to stay in the model (e.g. SL = 0.05)
    Step 2. Fit the full model with all possible preditors.
    Step 3. Consider the preditor with the highest p-value. If p>SL, go to Step 4
            Otherwise, goto FIN. 
            (FIN: Your model is ready)
    Step 4. Remove the preditor.
    Step 5. Fit model without this variable*
    
3. Forward Selection
    Step 1. Select a SL to enter the model (0.05)
    Step 2. Fit all simple regression models y~xn Select the one with the lowest p-value
    Step 3. Keep this variable and fit all possble models with one extra preditor added to the one(s)
    Step 4. Consider the preditor with the lowest p-value. If P<SL, goto Step 3, otherwise FIN.
            (FIN: Keep the previous model)
    
4. Bidirectional Elimination
    Step 1. Select a SL to enter and to stay in the model.
            e.g. SLenter = 0.05, SLstay = 0.05
    Step 2. Perform the next step of forward selection 
            (new variables must have p<SLenter to enter)
    Step 3. Perform ALL steps of backward elimination 
            (old variables must have p<SLstay to stay)
    Step 4. No new variables can enter and no old variables can exit
            (FIN: Your model is ready)
            
5. Score Comparison (All possible models)
    Step 1. Select a citerion of goodness of fit. (e.g. Akaike criterion)
    Step 2. Construct all possible regression models: 2^N-1 combinations
    Step 3. Select the one with the best criterion
    FIN. Your model is ready.
    (10 columns means 1023 models)
"""


# Linear Regeression Summary:
# Either simple linear or multiple linear, the code are the same.

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




'''--------------------------------Session 6--------------------------------'''

# Polynomial Linear Regression:
"""
    y = b0 + b1*x1 + b2*x1^2 + ... + bn*x1^n
    (Only one variable x1)
 
Q: Why called 'linear'?
A: refer to coefficients.
"""


'''--------------------------------Session 7--------------------------------'''

# SVR: Support Vector Regression
''' 
- SVMs that support linear and non-linear regression are SVR
- Instead of trying to fit the largest possiblr street between two classes while
    limiting margin violations, SVR tries to fit as many instances as possible 
    on the street while limiting margin violations;
- The width of the street is controlled by hyperparameter Epsilon;

- SVR performs a linear regression in a higher (dimensional space);
- We can think of SVR as if each data point in the training represents it's 
    own dimension, when you evaluate your kernel between your test point and the
    point in the training set, the resulting value gives you the coordinate of your 
    test point in that dimension;
- The vector we get when we evaluate the test point for all points in the training
    set, ->k is the representation of the test point in the higher dimensional 
    space;
- Once you have the vector, then use it to perform linear regression.

- It requires a training set: t = {X,Y} which covers the domain of interest and 
    is accompanied by solutions on that domain.
- The work of the SVM is to approximate the function we used to generate the 
    training set: F(X) = Y
    
    X are used to define a hyperplane that separate two different classes 
    in your soltion.
    
    These vectors are used to perform linear regression. The vectors closest 
    to the test point are referred to as support vectors. We can evaluate our
    function anywhere so any vectors could be closest to our test evaluation
    location.
'''

# Building a SVR
'''
1. Collect a training set t = {X,Y}
2. Choose a kernel and its parameters as well as any regularization needed.
3. Form the correlation matrix, K
4. Tranin your machine, exactly or approximately, to get contraction coefficients
    a = {ai}
5. Use those coefficientts, create your estimator f(X,a,x*) = y*
'''
# Choose a kernel & Regularization
''' Gaussion;;  Noise'''

# Correlation Matrix K
''' Ka = y
    y: the vector of values correponding to your training set,
    K: correlation matirx
    a: a set of unknowns we need to solve for
    
    So, a = K^(-1)y
    
Then use the coefficients we found during the optimization step and the kernel.
To estimate y* for a test point, x* compute the correlation vector k
y* = ak
'''

# SVR has a different goal with linear regression. 
# In linear regression we try to minimize the error between the presction and data.
# In SVR we make sure that error do not exceed the threshold.



'''--------------------------------Session 8--------------------------------'''
# Decision Tree Regression




'''--------------------------------Session 9--------------------------------'''
# Random Forest Regression

# Ensemble Learning
'''Random Forest is a version of Ensemble Learning.'''

# Random Forest Regression
"""
Step 1. Pick a random K data points from the Training set.
Step 2. Build the Decision Tree associated to these K data points.
Step 3. Choose the number Ntree of trees you want to build and repeat 1 & 2.
Step 4. For a new data point, make each one of your Ntree trees to predict Y,
        and assign the new data point across the average of all the predicted 
        Y values.
"""



'''-------------------------------Session 10--------------------------------'''
# R-squared
'''
Sum of Squares of residuals 残差平方和:
    SSres = sum(yi - yi`)^2
    yi` is the predicted value.
    
Total Sum of Squares 总平方和:  
    SStot = sum(yi - y_avg)^2
    y_avg is a constant showing a horizontal line in the datatset.


    R squared:
            R^2 = 1 - SSres/SStot
    It implies how good is your line compared to the average line.
    If SSres is 0, then R = 0 ----> ideal!
    R^2 - Goodness of fit (Greater is better)

The more variables we add, the larger R-squared will be.
So with the variables we have, it can never be decreased.
'''

# Adjusted R-squared
'''
Adjusted R^2:
    Adj_R^2 = 1 - (1 - R^2)*(n - 1)/(n - p - 1)
    
    p: number of regressors
    n: sample size
    

'''





