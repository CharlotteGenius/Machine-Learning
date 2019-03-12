# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values 
# Level column already shows the position difference, so no need to encode position
# And we want X to be matrix in machine learning, not a vector,
# so we still write as 1:2, not 1
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# 1. We only have 10 datasets, no need to split or test
# 2. We want accurate prediction, so we need more training data
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5) 
# The degree of the polynomial features. Default = 2.
# Try different degrees! It's fun.
X_poly = poly_reg.fit_transform(X)
# Now X has 1 variable, after poly transform, X_poly has 5 columns
# column 0 shows X^0 which is all 1
# column 1 shows X^1 which is original X itself ....
# And column 4 have X^4 values in X_poly.

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    # Don't use X_poly: easy to change & seperate these two functions.
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
    # arrange(start,stop,step)
    # Return evenly spaced values within a given interval. 
    # X_grid has 90 datasets.
X_grid = X_grid.reshape((len(X_grid), 1))
    # If we don't reshape(), X_grid is a vector rather than a matrix.
    # It will return error
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))