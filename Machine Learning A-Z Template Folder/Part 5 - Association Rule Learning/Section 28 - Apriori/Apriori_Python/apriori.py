# Apriori

# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, 7501):
    items = [str(dataset.values[i,j]) for j in range(0, 20)]
    # dataset.values[row,column]
    transactions.append(items)

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.25, min_lift = 4, min_length = 2)
# I tried different values, below is what's told in tutorial
'''
min_support = 0.003  --> We focus on items that were purchased at least 3 times a day, this dataset is over the whole week, tso he min_support is 3*7/7500 = 0.0028
min_confidence = 0.2  --> Good combination with 0.003 min_support
min_lift = 3  --> Try different values
min_length = 2  --> the minumum number of items that have relations
'''

# Visualising the result
results = list(rules)


# Usually it's done with one line above, I edited codes below to get better view.
# Bad-looking code, but nicer results :)
results_list = []
for k in range(0, len(results)):
    relatives = list(results[k][0])
    
    support = 'support='+str(results[k][1])
    confidence = str(results[k][2]).split(', ')[2]
    lift = str(results[k][2]).split(', ')[3][:-2]
    
    results_list.append([relatives,support,confidence,lift])
    # \t gives a tab space
    # \n starts a new line
