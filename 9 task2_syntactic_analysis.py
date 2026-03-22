#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Task 2: Syntactic Analysis and Preprocessing for Topic Modelling, saving processes data into nlp format for topic modelling

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd
# for lemmaztization
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_for_topics(text):
    # Cleaning: Lowercase and remove non-alphabetic characters like punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    
    # Stop words: Tokenization & Stop Word Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Lemmatization: Convert words to their base form
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(clean_tokens)

df = pd.read_csv('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/cleaned_social_media.csv')
df['nlp_ready_text'] = df['post'].apply(preprocess_for_topics)
df.to_csv('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/nlp_processed.csv', index=False)

"""
Feature,Original Post (in cleaned_social_media.csv),Processed Text (in nlp_ready_text column)
Punctuation, Wait! Are the vaccines safe?, wait vaccine safe
Capitalization, TRUMP wins in Michigan!, trump win michigan
Grammar, The police are investigating the claims, police investigate claim
"""