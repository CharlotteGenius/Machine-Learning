# Artificial Neural Network

# Installing Keras
# conda install -c conda-forge keras

#%% Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # notice it's 3-12 columns
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
'''encode countries (France, Spanish, Germany) to 0,1,2'''
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
'''encode gender (female, male) to 0,1'''
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
'''use dumming variables to avoid the relations between those numbers
    3 countries --> 2 digits dummy variables (00,01,10)'''
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
'''This way the three countries are encoded as 001,010,100
but we don't need the first number, because without first column, the countries are still encoded properly'''
X = X[:, 1:] # all the columns except the first one

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
'''add(layer)
Dense() takes care of Step 1. To initialize the weights to small numbers
output_dim is how many nodes you want to add in this layer, we use parameter tuning (and validation data) to find out this number. Usually the average of number of inputs and number of outputs could be good, which is (11+1)/2=6
init = 'uniform'  uniform function to initialize the weights
activation = 'relu'  is to use rectifier function in he first hidden layer
input_dim = 11  is the first input layer, 11 is the number of inputs
'''

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
'''
sigmiod fucntion for output layer
'''

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#%% Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
'''this gives us some float, we need to convert this into binary --> 1 or 0'''
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)