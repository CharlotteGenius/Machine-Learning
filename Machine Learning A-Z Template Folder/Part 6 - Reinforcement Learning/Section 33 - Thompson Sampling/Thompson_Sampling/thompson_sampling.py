# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []

'''Step1. At each round n, we consider two numbers for each ad i:
    - Ni1(n) the # of times the ad i got reward 1 up to round n,
    - Ni0(n) the # of times the ad i got reward 0 up to round n.
'''
numbers_of_rewards_1 = [0] * d # Ni1(n)
numbers_of_rewards_0 = [0] * d # Ni0(n)

total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        '''Step 2. For each ad i, we take a random draw from the distribution:
            θi(n) = β( Ni1(n)+1, Ni0(n)+1 )'''
        if random_beta > max_random:
            max_random = random_beta
            '''Step 3. We select the ad that has the highest θi(n)'''
            ad = i
    ads_selected.append(ad)
    
    # update two numbers
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    
    total_reward = total_reward + reward


# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()