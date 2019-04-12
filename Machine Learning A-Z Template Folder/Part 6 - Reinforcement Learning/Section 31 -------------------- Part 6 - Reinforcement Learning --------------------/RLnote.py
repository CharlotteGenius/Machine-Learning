#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:30:16 2019

@author: xiangyinyu
"""

# ================================= Part 6 ===================================
# Reinforcement Learning
'''
Online learning or Interactive learning;
Since it's dynamic according to the users' reactions.

E.g. After 10 rounds, according to 10 rounds result, we'll decide which ads it'll show to the user.
It depends on the observations at the beginning of the experiment up to the present time.
'''

'''----------------------------- Section 27 --------------------------------'''
# Upper Confidence Bound (UCB)

# the Multi-armed Bandit Problem
'''how to bet the best machine?
Combine exploration and exploitation of these machines in order to find the best distribution.

E.g. We have 500 ads design in our company, how do we know which one is the best?
We want to get results soon and with less money.


- We have d arms. E.g. arms are ads we display to users each time they connect to a web page.

- Each time a user connects to this web page, that makes a round.

- At each round n, we choose one ad to display to the user.

- At each roung n, ad i gives reward ri(n)∈{0,1}: ri(n)=1 if user clicked on the ad i, 0 if the user didn't.

- Our goal is to maximize the total reward we get over many rounds.



Step 1. At each round, we consider two numbers for each ad i:
    - Ni(n): # of times the ad i was selected upto round n
    - Ri(n): the sum of rewards of the ad i up to round n
    
Step 2. From these two numbers we compute:
    - the average reward of ad i up to round n: 
        ri(n) = Ri(n)/Ni(n)
    - the confidence interval [ri(n)-δi(n), ri(n)+δi(n)] at round n:
        δi(n) = sqrt{ 3log(n) / 2Ni(n) }
        
Step 3. Select ad i that has maximum UCB ri(n)+δi(n)
'''

