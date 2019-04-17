# Natural Language Processing
# Notice it's .tsv file in the directory
# csv - comma separated value; tsv - tab separated value
# In the reviews, there're some sentences with comma, computer will not recognize it properly

'''
fn+F5 run the whole file
fn+F9 run selection or current line
commad+enter run the cell
'''

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# quoting = 3  is to ignore double quote

#%% Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # ^ means what not to remove
    # ' ' means removed character will be replaced with a space, avoiding two parts stuck together
    review = review.lower().split()
    
    ps = PorterStemmer() # 词干提取 e.g. 'loved'->'love'; 'loving'->'love'
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#%% Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# extract 1500 most frequent words, you can try 1000 or others
'''
This function contains the functions shown in the cell above.
if CountVectorizer(stop_words = 'english'):
It will automatically remove the stop words in 'english'

if CountVectorizer(token_pattern = '[^a-zA-Z]'):
token_pattern is what you want to keep.

It has many variables to avoid cleaning the texts step.
But it's always good to do it yourself because it avoids mistakes and you're in control.
'''
X = cv.fit_transform(corpus).toarray() # to create a sparse matrix
y = dataset.iloc[:, 1].values # .value to transfer series to integers

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#%% Try multiple methods in previous sections and choose one with cm
# Methods usually used in NLP are:
# Naive Bayes, Decision Tree and Random Forst
# Try even other classification models that we haven't covered in Part 3 - Classification. Good ones for NLP include:
# CART
# C5.0
# Maximum Entropy

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# =============================================================================
# # Fitting Decision Tree Classification to the Training set
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)
# =============================================================================

#%% Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#%% Evaluate the performance of each of these models
'''
TP: true positive
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 Score = 2 * Precision * Recall / (Precision + Recall)
'''
Accuracy = (cm[0,0]+cm[1,1]) / len(X_test)
Precision = cm[0,0] / (cm[0,0]+cm[1,0])
Recall = cm[0,0] / (cm[0,0]+cm[0,1])
F1_Score = 2 * Precision * Recall / (Precision + Recall)

