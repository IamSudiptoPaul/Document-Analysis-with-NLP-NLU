#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Score Comparison of BoW, TF-IDF, and BGE-M3 Semantic Embeddings

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from FlagEmbedding import BGEM3FlagModel

df = pd.read_csv('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/nlp_processed.csv').dropna(subset=['nlp_ready_text', 'class_label']).sample(2000, random_state=42)
X_text, y = df['nlp_ready_text'], df['class_label']

# Load Models
v_bow = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_bow.pkl')
v_tfidf = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_tfidf.pkl')
bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Simple Evaluation Function
def show_scores(X_vec, y_true, name):
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_true, test_size=0.2, random_state=42)
    y_pred = LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test)
    print(f"\n--- {name} Results ---\n", classification_report(y_test, y_pred))

# Show Results
show_scores(v_bow.transform(X_text), y, "Bag of Words")
show_scores(v_tfidf.transform(X_text), y, "TF-IDF")

print("Processing BGE-M3 Semantic Embeddings:")
X_sem = bge_model.encode(X_text.tolist())['dense_vecs']
show_scores(X_sem, y, "BGE-M3 Semantic")



"""
Feature,Bag-of-Words (BoW),TF-IDF,BGE-M3 (Embeddings)
Logic,Simple word frequency (counts),Frequency weighted by rarity,Semantic context & meaning
NLU Level,Syntactic (Surface-level),Statistical (Importance),Semantic (Deep Understanding)
Doctor vs Physician,Seen as 2 different things,Seen as 2 different things,Seen as nearly identical
Handling Noise,Highly sensitive to stop words,Good at ignoring common words,Context-aware (ignores noise)
Best For...,"Quick baseline, simple tasks",Keyword-rich topic discovery,"Nuanced, intent-based analysis"
Data Sparsity,High (mostly zeros),High (mostly zeros),Dense (1024 numeric values)
"""