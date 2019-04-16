# Upper Confidence Bound
# UCB

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
'''In the file, they're series of 1s and 0s 
showing 10000 users reacts to 10 different ads'''

# Implementing UCB
import math
N = 10000
d = 10
# 10000 users are showed 10 ads per round
ads_selected = []

'''Step 1. Consider two variables Ni(n) and Ri(n)
Ni(n) the # of times that ad i was selected at each round n
Ri(n) sum of rewards of the ad i up to round n'''
numbers_of_selections = [0] * d # Ni(n)
sums_of_rewards = [0] * d       # Ri(n)
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            '''Step 2. From these two numbers we compute:
                - the average reward of ad i up to round n: 
                     ri(n) = Ri(n)/Ni(n)
             - the confidence interval [ri(n)-δi(n), ri(n)+δi(n)] at round n:
                     δi(n) = sqrt{ 3log(n) / 2Ni(n) }
            '''
            average_reward = sums_of_rewards[i] / numbers_of_selections[i] # ri(n)
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]) # δi(n)
            upper_bound = average_reward + delta_i
            '''Step 3. Select ad i that has maximum UCB ri(n)+δi(n)
            '''
        else:
            upper_bound = 1e400 # just choose a very large number
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            # if larger, set max equal to it, if smaller, keep the last upper bound
            ad = i
            # until then, ad stores the max-ucb ad number i
    ads_selected.append(ad)
    # ads_selected stores every max-ucb ad in each round
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    # update the # of times that ad i was selected
    reward = dataset.values[n, ad] # [row, column]; 0 or 1
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
