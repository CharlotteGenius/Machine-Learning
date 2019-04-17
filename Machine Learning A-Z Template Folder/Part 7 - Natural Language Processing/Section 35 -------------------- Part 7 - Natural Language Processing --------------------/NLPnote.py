#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:01:59 2019

@author: xiangyinyu
"""

# If you wanna import this file
# Move triple quatation marks at the end of the file.

# ================================= Part 7 ===================================
# Natural Language Processing (NLP)

Natural Language Processing (or NLP) is applying Machine Learning models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of Natural Language Processing. Whenever you dictate something into your iPhone / Android device that is then converted to text, that’s an NLP algorithm in action.

You can also use NLP on a text review to predict if the review is a good one or a bad one. You can use NLP on an article to predict some categories of the articles you are trying to segment. You can use NLP on a book to predict the genre of the book. And it can go further, you can use NLP to build a machine translator or a speech recognition system, and in that last example you use classification algorithms to classify language. 
Speaking of classification algorithms, most of NLP algorithms are classification models, and they include Logistic Regression, Naive Bayes, CART which is a model based on decision trees, Maximum Entropy again related to Decision Trees, Hidden Markov Models which are models based on Markov processes.


In this part, you will understand and learn how to:

- Clean texts to prepare them for the Machine Learning models,
- Create a Bag of Words model,
- Apply Machine Learning models onto this Bag of Worlds model.


'''----------------------------- Section 36 --------------------------------'''

# Main NLP library examples:

- Natural Language Toolkit - NLTK
- SpaCy
- Stanford NLP
- OpenNlp

https://www.nltk.org
https://www.tensorflow.org/tutorials/word2vec



# NLP - Bag of words

A very well-known model in NLP is the Bag of Words model. It is a model used to preprocess the texts to classify before fitting the classification algorithms on the observations containing the texts.

It involves two things:
    1. A vocabulary of known words
    2. A measure of presence of known words

