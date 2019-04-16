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

'''----------------------------- Section 32 --------------------------------'''
# Upper Confidence Bound (UCB)

# the Multi-armed Bandit Problem
'''How to bet the best machine?
Combine exploration and exploitation of these machines in order to find the best distribution.

E.g. We have 500 ads design in our company, how do we know which one is the best?
We want to get results soon and with less money.

the Multi-armed Bandit Problem is:
    - We have d arms. For example, arms are ads we display to users each time they connect to a web page.
    - Each time a user connects to this web page, that makes a round.
    - At each round n, we choose one ad to display to the user. (Always display/exploit the best one)
    - At each round n, ad i gives reward ri(n)∈{0,1}: ri(n)=1 if user clicked on the ad i, 0 if the user didn't.
    - Our goal is to maximize the total reward we get over many rounds.
'''

# Upper Confidence Bound (UCB) Algorithm
'''
Step 1. At each round, we consider two numbers for each ad i:
    - Ni(n): # of times the ad i was selected upto round n
    - Ri(n): the sum of rewards of the ad i up to round n
    
Step 2. From these two numbers we compute:
    - the average reward of ad i up to round n: 
        ri(n) = Ri(n)/Ni(n)
    - the confidence interval [ri(n)-δi(n), ri(n)+δi(n)] at round n:
        δi(n) = sqrt{ 3log(n) / 2Ni(n) }
        
Step 3. Select ad i that has maximum UCB ri(n)+δi(n)

先对每一个臂都试一遍
之后，每次选择UCB值最大的那个臂
其中ri是这个臂到目前的收益均值，δi叫做bonus
本质上是均值的标准差，n是目前的试验次数

这个公式反映：均值越大，标准差越小，被选中的概率会越来越大，起到了exploit的作用；
同时哪些被选次数较少的臂也会得到试验机会，起到了explore的作用。

'''


'''----------------------------- Section 33 --------------------------------'''
# Thompson Sampling
# a probabilistic algorithm

'''
Step1. At each round n, we consider two numbers for each ad i:
    - Ni1(n) the # of times the ad i got reward 1 up to round n,
    - Ni0(n) the # of times the ad i got reward 0 up to round n.

Step 2. For each ad i, we take a random draw from the distribution  below:
    θi(n) = β( Ni1(n)+1, Ni0(n)+1 )

Step 3. We select the ad that has the highest θi(n)
'''

'''
https://zhuanlan.zhihu.com/p/21388070

假设每个臂是否产生收益，其背后有一个概率分布，产生收益的概率为p
我们不断地试验，去估计出一个置信度较高的*概率p的概率分布*就能近似解决这个问题了。
怎么能估计概率p的概率分布呢？ 
答案是假设概率p的概率分布符合beta(win, lose)分布，它有两个参数: win, lose。
每个臂都维护一个beta分布的参数。每次试验后，选中一个臂，摇一下，有收益则该臂的win增加1，否则该臂的lose增加1。
每次选择臂的方式是：用每个臂现有的beta分布产生一个随机数b，选择所有臂产生的随机数中最大的那个臂去摇。
'''

# code
choice = numpy.argmax(pymc.rbeta(1 + self.wins, 1 + self.trials - self.wins))



