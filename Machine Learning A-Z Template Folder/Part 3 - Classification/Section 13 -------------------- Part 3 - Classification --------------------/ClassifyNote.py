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
        accoding to the Euclidean distance
Step 3. Among these K neighbors, count the number of data points in each category
Step 4. Assign the new data point to the category 
        where you counted the most neighbors.

Notice: K is the number of neighbors you choose, Not A Radius!!
'''


'''---------------------------------Section 16-------------------------------'''
# SVM
'''It's to find the best fitting boundary which helps us to separate different classes.

Say we have two classes data with their margin points.
And we use these margin points to find two parallels that forms a gap between twwo classes.
- And this gap is the Maximum Margin, and these points we call Suppport Vectors.

- The line in the middle of this gap is called Maximum Margin Hyperplane.
(or Maximum Margin Classfier)

- The two margin lines are called Positive Hyperplane and Negative Hyperplane, respectively.
'''

# What's so special about SVM?





'''---------------------------------Section 17-------------------------------'''
# Kernel SVM
'''
In the last case, we have panel of points that can be seperated with a linear line easily.
Then what about other situations like,
some points are surrounded by other points, (Non linear models)?

So SVM is to use these support vectors to find the proper boundary (Assumption)
'''


# Higher-Dimension Space
'''
For example, in one-dimension space, we have:
    ---------- ○ ○ ○ ○ ● ● ● ○ ○ ○ ----->
    
It's not linear in the line, and we want to separate them linearly.
We can made some computaions with these variables like f = (x-5)^2
Then these dots are like balls, bounce on to 2-dimensional plane.
And then we can separate them with a line.

Same thing to 2-D space:
    F1(x1,x2) ==============>> F2(x1,x2,z)
              Mapping function

z is the new dimension, and we can separate points using hyperplane.

After that, we need to project them back into 2-D:
    F2(x1,x2,z) ==============>> F1(x1,x2)
                  Projection
In 2-D, It still looks like non-linear separator.
'''

# Mapping to higher dimension can be really Compute-Intensive!
# Kernel Trick doesn't require mapping.

# Kernel Rules
''' K(-) = 1   if condition
           0   otherwise
'''


# The Gaussian RBF kernel
'''
    K(x,li) = e^(-|x-li|^2 / 2σ^2)
                     ^
                  Distance^2

σ↑ --> width↑
'''

# Type of Kernel Functions
'''
Gaussian
Sigmoid: K(X,Y) = tanh(v.X`Y+r) ---> 'S' model
Polynomial
'''




'''---------------------------------Section 18-------------------------------'''
# Naive Bayes

# Baye's Theorem
'''
P(A|B) = P(B|A) * P(A)/P(B)
'''

# Naive Bayes
'''
Previously, we have datasates of class1 & class2,
When we have a new point X, how to define its class?

Step 1. P(c1|X) = P(X|c1) * P(c1)/P(X)
    (1). P(c1): Prior Probability:
        P(c1) = (# of c1) / (total observations)
        
    (2). P(X): Marginal Likelihood:
        P(X) = (# of Similar observations with X) / (total observations)
        we can draw a circle with X as the origin
        and the points in the circle are similar observations
    
    (3). P(X|c1): Likelihood:
        We know that: P(a|b) = P(a∩b)/P(b)
        Draw this circle again, and count the number of c1 in this circle
        P(X|c1) = (# of c1 in the circle) / (# of c1)
        
    (4). P(c1|X): Posterior Probability:
        Get result according to Baye's Theorem

Step 2. P(c2|X) = P(X|c2) * P(c2)/P(X)

Step 3. P(c1|X) v.s. P(c2|X)
    If P(c1|X) > P(c2|X), then X will be classified as class 1.
'''

# Why 'Naive'?
'''
We assume that they're all independent variables.
But actually there's always some correlation between two variables.
'''

# P(X)
'''
In Step 1 and Step 2, we both calculate P(X)
While in both aituations, it's the same value
So we don't need to calculate it everytime.
Just need to compare:
    P(X|c1)*P(c1) v.s. P(X|c2)*P(c2)
'''

# More than 2 classes?
'''
When it's only 2 classes, we actually don't need to calculate the second probability.
If we get 0.8 in Step 1, then obviouly, Step 2 we'll get 0.2.

If we have more classes, we probably need to calculate more steps.
'''


'''---------------------------------Section 19-------------------------------'''
# Decision Tree Clasification


























