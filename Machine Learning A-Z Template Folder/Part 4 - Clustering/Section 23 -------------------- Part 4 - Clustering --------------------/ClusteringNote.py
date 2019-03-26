#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:10:52 2019

@author: xiangyinyu
"""


# ================================= Part 4. ===================================
# Clustering
# cluster簇: a group of similar things or occuring closely together. (Congregate)
'''
Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.

In this part, you will understand and learn how to implement the following Machine Learning Clustering models:

    1. K-Means Clustering
    2. Hierarchical Clustering
'''


# Supervised learning: 
'''    suppose you have a basket and it is filled with some fresh fruits and your task is to arrange the same type fruits at one place. suppose the fruits are apple,banana,cherry, and grape. so you already know from your previous work that, the shape of each and every fruit so it is easy to arrange the same type of fruits at one place. here your previous work is called as trained data in data mining. so you already learn the things from your trained data, This is because of you have a response variable which says you that if some fruit have so and so features it is grape, like that for each and every fruit.

This type of data you will get from the trained data. This type of learning is called as supervised learning. This type solving problem comes under Classification. So you already learn the things so you can do you job confidently.'''

# unsupervised: 
'''    suppose you have a basket and it is filled with some fresh fruits and your task is to arrange the same type fruits at one place.

This time you don't know any thing about that fruits, you are first time seeing these fruits so how will you arrange the same type of fruits.

What you will do first is you take on the fruit and you will select any physical character of that particular fruit. suppose you taken color.

Then you will arrange them based on the color, then the groups will be some thing like this. RED COLOR GROUP: apples & cherry fruits. GREEN COLOR GROUP: bananas & grapes. so now you will take another physical character as size, so now the groups will be some thing like this. RED COLOR AND BIG SIZE: apple. RED COLOR AND SMALL SIZE: cherry fruits. GREEN COLOR AND BIG SIZE: bananas. GREEN COLOR AND SMALL SIZE: grapes.'''



'''----------------------------- Section 24 --------------------------------'''
# K-means Clustering
'''
Step 1. Choose the number K of clusters
Step 2. Select at RANDOM K points, the centroids (not necassarily from your dataset)
Step 3. Assign each data point to the closest centroid ==> That forms K clusters
Step 4. Compute and place the new centroid of each cluster
Step 5. Reassign each data point to the new closest centroid.
        If any reassignment took place, goto Step 4, otherwise goto FIN.
FIN. Your model is ready


Notice:
    'closest' in Step 3 implies distance, while there're several different distance types you should decide.
'''
   
# K-means clustering initialization trap
# random initialization trap
'''
The selection of the central in Step 2 can potentially dictate the outcome of the algorithm.
        ||
        \/
K-Means++ algorithm --> check it out online!
'''

# Choose the right number K of clusters
# WCSS:
'''
Start with K=1, all the points belong to 1 cluster
And we compute the WCSS (Suppose it's 8000)

And we increase K to 2, and WCSS will decrease...(Suppose it's 3000)

we increase K to 3, and WCSS will decrease...(Suppose it's 1000)

Suppose we have 50 points, the largest K value is 50, and the corresponding WCSS will be 0.
So WCSS is between 0 ~ 8000(when K=1)
'''

# So how do we choose K? Elbow Method
'''
We can draw a plot with WCSS - K
And for example, from 8000 to 3000 and then 1000, they are large drop
but after k=3, WCSS decrease slowly...
And it's obvious from the curve that k=3 is the elbow, so we can choose k=3
it's the optimal number of clusters.

While sometimes it's not that obvious.
You should decide what to use. 
Or if you're not sure, you can do more tests with different Ks and see the differences and then decide a better K for your model.
'''

















