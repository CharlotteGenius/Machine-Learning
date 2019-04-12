#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:11:39 2019

@author: xiangyinyu
"""

# ================================= Part 5 ===================================
# Association Rule Learning
'''
how to implement the following Association Rule Learning models:

1. Apriori
2. Eclat
'''

'''----------------------------- Section 24 --------------------------------'''
# Apriori Rule Learning
'''
ARL: it's about the rules/connections between two or more things.
Like, somebody did something who also did another thing.
Althogh sometimes these two things don't have any obvious connection.

For example, people bought diaper also bought beer.
'''

# ARL - Movie recommendation
'''
For example, a chart record customers and the movies they watch.
We can see people who watched movie 1 also watched movie 2.
Then movie 1 and movie 2 have some rules between them.
'''

# ARL - Market basket Optimization
'''
For example, people who bought burgers also bought fries, et al.
'''

# Apriori - Support
# Apriori - Confidence
# Apriori - Lift
'''
Step 1.
We have 100 people in total, and 10 people watched movie 2, then,
Apriori support movie 2: Support(M2) = 10/100 = 10%

Step 2.
We have an assumption that movie 1 and 2 have connections.
We have 40 people watched movie 1, and in 40, 7 people watched movie 2.
So we want to check with Confident(M1->M2) = 7/40 = 17.5%

Step 3.
Lift(M1->M2) = Confidence(M1->M2) / Support(M2)
             = 17.5% / 10%
             = 1.75
It shows the likelihood that people like movie 1 also like movie 2
'''

# Apriori Algorithm
'''
Step 1. Set a minumum support and confidence
Step 2. Take all the subsets in transactions having higher support than minimum support
Step 3. Take all the rules of these subsets having higher confidence
Step 4. Sort the rules by decreasing lift
'''



'''----------------------------- Section 25 --------------------------------'''
# Elact (trivial)
'''We only have support in Elact model, and this is the same with step 1 in Apriori model.
So we only look at how popular this movie is.
But that's too weak, so we can calculate support of two or more items,
for example, how popular are movie 1 or movie 2 together.
so it shows the likelihood that these two movies happen together.
'''













