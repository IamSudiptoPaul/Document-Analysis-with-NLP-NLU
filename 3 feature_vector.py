#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Feature Vectorization: This script reads the cleaned social media posts, transforms them into a numerical format using TF-IDF vectorization, and 
saves the processed data for use in the Neural Network model.

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd
# Using Sklearn's built-in library pickle for saving the processed data
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/cleaned_social_media.csv')

# I am transforming the text into a numerical representation with 5000 features
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['post'])
y = df['class_label'].astype(int).values

# I am splitting the data to reserve a test set for generalization checks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# processed_data.pkl file contains the actual numerical data ready for the Neural Network. (X_train, X_test, y_train, y_test)
with open('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/processed_data.pkl', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

# vectorizer.pkl contains unseen data and mathematical weights assigned to each word
with open('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Processing complete.")