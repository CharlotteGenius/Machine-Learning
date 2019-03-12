# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # Independet Variables
y = dataset.iloc[:, 4].values # Profit: Dependent Variables

# Encoding categorical data
# Encoding independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3]) # state catagory is the only variable to encode
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (Library will do it for us) 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




# Building the optional model using Backward Eliminate ========================
# Backward Elimination with only p-values:

'''---------------------Step 1. we choose SL = 0.05----------------------'''

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# This Adds a column of ones at the beginning in X matrix
# Corresponding to the equation y = b0*1 + b1*x1 + b2*x2 .....
# Here, we add a column of ones,
# Because in statsmodels, we'll need the first variable X0 to deal with b0.


'''----------------------Step 2. fit the full model----------------------'''
X_opt = X[:,[0,1,2,3,4,5]]

# Create a OLS regressor, Simple Ordinary Squares Model
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# OLS(endog, exog, missing, hasconst)
# endog: 1-d endogenous response variable, the DV.
regressor_OLS.summary()

""" After excuting 5 lines above, we get:
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================

x1,x2 are two dummy variables for States, 
x3 is R&D spend, x4 is Admin spend, x5 is marketing spend

We look for highest p-value, which is 0.990
and it's way more than SL = 0.05, so we need to remove this preditor which is x2.

"""

'''--------------Step 5. Fit model without this variable*------------------'''
X_opt = X[:,[0,1,3,4,5]] # remove x2

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

""" After excuting 3 lines above, we get:
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
highest p-value is 0.940 > 0.05 from x1 in this chart,
x1 here is actually from index 1, which is state dummy variable, and remove it again.
"""

X_opt = X[:,[0,3,4,5]] # remove x1

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

""" After excuting 3 lines above, we get:
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
highest p-value is 0.602 > 0.05, so we remove x2, which is index 4.
"""

X_opt = X[:,[0,3,5]] # remove x4

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

""" After excuting 3 lines above, we get:
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
0.06>0.05
"""
X_opt = X[:,[0,3]] # remove x5

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
Good. Model is ready.
"""
# This result shows that only R&D spend has strong relativeness with profit.
# Predicting the Test set results
y_pred_opt = regressor_OLS.predict(X_test[:,[0,3]])
# This actually shows bad results.

# Plus, I just learned that I could use max(regressor_OLS.pvalues)
# No need to excute every time to look for max p-values...
# So the condition would be:
''' if max(regressor_OLS.pvalues) > SL: remove variable and continue  '''


# Backward Elimination with p-values and Adjusted R Squared:

def backwardElimination(x, sl):
    # x: model inputs
    numVars = len(x[0])
    temp = np.zeros((50,6))
    
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        adjR_before = regressor_OLS.rsquared_adj
        
        if maxVar > sl:
            j = list(regressor_OLS.pvalues).index(maxVar)
            # find the variable j that has max p-value
            temp[:,j] = x[:, j]
            # temp to restore the j column, in case we need to add it back
            x = np.delete(x, j, 1) # 1 means delete column
            tmp_regressor = sm.OLS(y, x).fit()
            adjR_after = tmp_regressor.rsquared_adj
            
            if adjR_before >= adjR_after:
                # if it's better that not removing this variable
                # then add this variable back and 
                # the model is ready with current variables
                x_rollback = np.hstack((x, temp[:,[0,j]]))
                x_rollback = np.delete(x_rollback, j, 1)
                print(regressor_OLS.summary())
                return x_rollback
    
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0,1,2,3,4,5]]
# Notice that X here is already added the first ones column!
X_Modeled = backwardElimination(X_opt, SL)
