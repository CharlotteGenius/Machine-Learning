#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:41:35 2019

@author: xiangyinyu
"""

# Classification
'''
Unlike regression where you predict a continuous number, you use classification to predict a category. 
There is a wide variety of classification applications from medicine to marketing. Classification models include linear models like Logistic Regression, SVM, and nonlinear ones like K-NN, Kernel SVM and Random Forests.

Machine Learning Classification models:

1. Logistic Regression
2. K-Nearest Neighbors (K-NN)
3. Support Vector Machine (SVM)
4. Kernel SVM
5. Naive Bayes
6. Decision Tree Classification
7. Random Forest Classification
'''

'''---------------------------------Section 14-------------------------------'''
# Logistic Regression
'''
Start with a model that:
    For some points, the output is 'yes' while others are 'no'.
    And we interpret this model with '1' and '0'
    So the gragh we got from these inputs are located on line 0 and line 1.
    That means, in some case, there's a large possibility that the result is 0
    and in other case, the result is very possible to be 1.
    
    It's improper to use linear regression in this model.
    So we applied this formation here:

        Linear model ==> Sigmoid Function ==> Logistic Function
        y = b0+b1*x  ==> p = 1/(1+e^(-y)) ==> ln(p/(1-p)) = b0+b1*x

And the shape of logistic function is an 'S'
The x axis is the input variable and y axis is probability.
So clearly, probability is between 0 and 1.

Say x = 20 and in the model we get p = 0.7
According to the model, there's probability of 70% that x will result in 'yes'.

But this result is probability, can we predict actual DV --> y?
'''

# We set a line at 0.5
'''
if p less than 50%, then we say y` is 'no', 
and p larger than 0.5, we say y` is 'yes'.
When we have to choose a value between 1 and 0.
'''

# Logistic model is also a linear model.
# We build Logistic Model just to better fit this model. 
# And the line we set, could higher or lower.




'''---------------------------------Section 15-------------------------------'''
# K-Nearest Neighbors

'''
Before KNN, we have two catagories c1 and c2, 
and we have a new point p, how to we know it belongs to c1 or c2?


Step 1. Choose the number K of neighbors.
Step 2. Take the K nearest neighbors of the new data point, 
        accoding to the Euclidean dstance
Step 3. Among these K neighbors, count the number of data points in each category
Step 4. Assign the new data point to the category 
        where you counted the most neighbors.


























