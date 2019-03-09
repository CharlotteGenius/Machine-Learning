# Data preprocessing

# Importing the libriries ====================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset ======================================================
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values  
# matrix features [row, column]
# for the row: take all the items
# for the column: take all items except the last one

Y = dataset.iloc[:, 3].values
# take all the row, the 3rd column

# Type of X and Y are 'object', not list.

# Take care of missing data  ==================================================
# if you can't see the full array, just run
# np.set_printopt ions(threshold = np.nan)

from sklearn.preprocessing import Imputer
# sikat learn contains librires to make machine learning models
# this allows to take care of missing data

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# for missing values encoded as np.nan (in dataset), use the string value "NaN"
# 均值"mean": replace using the the mean value along the axis
# 中位数"medium": using the medium
# 众数"most_frequent": using the most frequent value along the axis
# axis = 0 implies columns; axis = 1 implies rows

imputer = imputer.fit(X[:, 1:3])
# take all the rows and column index 1 and 2
# make imputer fit in to data X
X[:, 1:3] = imputer.transform(X[:, 1:3])
# replace the missing data

# For the two lines 36 and 40, I tried using this in one line:
"""X[:, 1:3] = imputer.fit_transform(X[:, 1:3])"""
# And it worked too.


# Encoding catagorical data ==================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# create an object in this class
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Encode makes the first column become their corresbonding code:
# France is 0, Spain is 2 and Germany is 1:
'''
So X became:
array([[0, 44.0, 72000.0],
       [2, 27.0, 48000.0],
       [1, 30.0, 54000.0],
       [2, 38.0, 61000.0],
       [1, 40.0, 63777.77777777778],
       [0, 35.0, 58000.0],
       [2, 38.77777777777778, 52000.0],
       [0, 48.0, 79000.0],
       [1, 50.0, 83000.0],
       [0, 37.0, 67000.0]], dtype=object)
'''
# But if it's like this, the computer will think 2 is greater than 1,
# So Spain is greater than Germany or so, but this doesn't make sense.
# So we wanna use 
'''Dummy Encoding'''
# use 3 columns to represent 3 countries and each column represent one.
# So we use OneHotEncoder library.


onehotencoder = OneHotEncoder(categorical_features = [0])
'''categorical_features = [0]'''
# This is to set which column is treated as categorical, now we say column 0,
# while default of this parameter is to treat all as categorical
X = onehotencoder.fit_transform(X).toarray()

# Encode another catagory Y 
# we only need to use label encoder sicne 1 implies 'yes' and 0 implies '0'
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into the Train set and Test set =======================
from sklearn.model_selection import train_test_split
X_train, X_test, Y_tarin, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# test_size = 0.2 implies 20% of data will be testing sets and the remaining 80% will the training sets.


# Feature Scaling =============================================================
# Salary and Age are in different scale which are 10k or 10, 
# This way, the difference (Euclidean Distance) could be very large,
# Or say, one variable is dominate to another,
# which is not acceptable in python. (too slow)
# So it's better to put variables (features) in a same scale (range)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
# ----------------- they are in a same scale now, the values changed like:
'''-1	2.64575	 -0.774597	0.263068 	0.123815'''
# We don't need to apply to Y, it only has 0 and 1.









