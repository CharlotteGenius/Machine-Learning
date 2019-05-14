#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:20:55 2019

@author: xiangyinyu
"""

# ================================= Part 8 ===================================
# Deep Learning
'''
Deep Learning is the most exciting and powerful branch of Machine Learning. Deep Learning models can be used for a variety of complex tasks:

    - Artificial Neural Networks for Regression and Classification
    - Convolutional Neural Networks for Computer Vision
    - Recurrent Neural Networks for Time Series Analysis
    - Self Organizing Maps for Feature Extraction
    - Deep Boltzmann Machines for Recommendation Systems
    - Auto Encoders for Recommendation Systems

In this part, you will understand and learn how to implement the following Deep Learning models:
    - Artificial Neural Networks for a Business Problem
    - Convolutional Neural Networks for a Computer Vision task
'''

# Geoffrey Hinton: Father of Deep Learning

'''----------------------------- Section 39 --------------------------------'''
# In this Section we'll learn:
    # The Nueron
    # The Activation Function
    # How do Nueral Networkd work?
    # How do Nueral Networks learn?
    # Gradient Descent
    # Stochastic Gradient Descent
    # Backpropagation

# The Nueron
    dendrite; axon
    
# The Activation Function
    1. Threshold Function:
        1 for x>0; 0 for x<0 (yes or no)
    2. Sigmoid Function:
        1/(1+e^(-x))
    3. Rectifier:
        max(x,0)
    4. Hyperbolic Tangent (tanh):
        (1-e^(-2x))/(1+e^(-2x))


# How do Nueral Networkd work?

Backpropagation:
        Cost Function implies difference between y & ŷ (Actuall & Predicted)
        An example of cost function: C = ∑ (1/2) * (ŷ - y)²
        
        With a series of weights we get a cost function and then go back to adjust the weights and run again, to minimize the cost function.

# Gradient Descent 梯度下降法/最速下降法
"Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model."
" 要使用梯度下降法找到一个函数的局部极小值，必须向函数上当前点对应梯度的反方向的规定步长距离点进行迭代搜索。如果相反地向梯度正方向迭代进行搜索，则会接近函数的局部极大值点；这个过程则被称为梯度上升法。"

# Curse of Dimensity

FLOPS: floating operation per second
Sunway Taihulight is the fastest computer in the world and can work 93 PFLOPS

if the function is not convex, then we might find a local minimum cost function.:

# Batch Gradient Descent
Adjust w after all rows
# Stochastic Gradient Descent
'''
What if the function is not convex?
Batch GD is to take all the rows of batch data to train the neural network and get the weights and adjust weights according to the cost function. 
while SGD is to take a first row, train the neural network and adjust the weights, then take the second row and do that again, 
We're adjusting weights after every single row rather than running all the rows.

SGD is more likely to find the global minimum rather than local minimum.
And, it's faster.

Another method is mini_batch GD.
'''

# Training the ANN with SGD
'''
Step 1. Randomly initialize the weights to small numbers close to 0 (but not 0).

Step 2. Input the first observation of your dataset in the input layer, each feature in one input node.

Step 3. Forward-Propagation: from the left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted resultly.

Step 4. Compare the predicted result to the actual result. Measure the generated error.

Step 5. Back-propagation: from right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decided by how much we update the weights.

Step 6. Repeat Steps 1-5 and update the weights after each observation (Reinforcemrnt Learning). Or: Repeat steps 1-5 but update the weights only after a batch of observations (Batch Learning).

Step 7. When the whole training set passed thru the ANN, that makes an epoch. Redo more epochs.

'''




'''----------------------------- Section 40 --------------------------------'''
# Convolutional Neural Network




































