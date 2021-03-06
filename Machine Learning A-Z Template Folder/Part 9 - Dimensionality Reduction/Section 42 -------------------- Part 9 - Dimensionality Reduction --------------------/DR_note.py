#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:48:17 2019

@author: xiangyinyu
"""


# ================================= Part 9 ===================================
# Dimensionality Reduction
'''
Remember in Part 3 - Classification, we worked with datasets composed of only two independent variables. We did for two reasons:
    1. Because we needed two dimensions to visualize better how Machine Learning models worked (by plotting the prediction regions and the prediction boundary for each model).
    2. Because whatever is the original number of our independent variables, we can often end up with two independent variables by applying an appropriate Dimensionality Reduction technique.


There are two types of Dimensionality Reduction techniques:
    1. Feature Selection
    2. Feature Extraction


Feature Selection techniques are:
    Backward Elimination, Forward Selection, Bidirectional Elimination, Score Comparison and more. 
 We covered these techniques in Part 2 - Regression.


In this part we will cover the following Feature Extraction techniques:
    1. Principal Component Analysis (PCA)
    2. Linear Discriminant Analysis (LDA)
    3. Kernel PCA
    4. Quadratic Discriminant Analysis (QDA)
'''




'''----------------------------- Section 43 --------------------------------'''
# Principal Component Analysis (PCA) 主成分分析法
'''
- Can be used in:
    - Noise filtering
    - Visulization
    - Feature Extraction
    - Stock market predictions
    - Gene data analysis

- The goal of PCA:
    - Identify pattern in data
    - Detect the correlations between variables
'''

# in a few words
'''
From the m independent variables of your dataset, PCA extracts p<=m new independent variables that explain the most the variance of the dataset, regardless of the dependent variable.

--> The fact that the DV is not considered makes PCA an unsupervised model.

'''


'''----------------------------- Section 43 --------------------------------'''
# Linear Discriminant Analysis (LDA) 线性判别分析
'''
LDA differs because in addition to finding the component axises with LDA we are interested in the axises that maximize the separation between multiple classes.


Breaking it down further:
    The goal is to project a feature space (a dataset n-dimension samples) onto a small subspace k (where k<=n-1) while maintaining the class-discriminatory information.
    
    Both PCA and LDA are linear transformation techniques used for dimensional reduction. PCA is described as unsupervised but LDA is supervised because of the relation to the dependent variable.
'''


# in a few words
'''
From the n independent variables of your dataset, PCA extracts p<=n new independent variables that separate the most the classes of the dependent variable.

--> The fact that the DV is considered makes LDA an supervised model.

'''






