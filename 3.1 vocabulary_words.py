#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Vocabulary Words: This script loads the TF-IDF vectorizer from the previous step, extracts the list of vocabulary words (features), and prints them out. 
This helps to understand which words are being used as features in the model.

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pickle

# Load bridge file
with open('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Get the list of all 5,000 words in the vocabulary
feature_names = vectorizer.get_feature_names_out()

# Print the first 20 words to see what they look like
print("Top Vocabulary Words:")
print(feature_names[:20])

# Can also search for a specific word to see if it's in vector
word_to_check = "government"
if word_to_check in feature_names:
    print(f"'{word_to_check}' is a vector word at index {list(feature_names).index(word_to_check)}")