#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Bag of Words (BoW) and TF-IDF Representation with LDA Topic Modelling

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/nlp_processed.csv').dropna(subset=['nlp_ready_text'])

def discover_topics(vectorizer_type, name):
    print(f"\nExperimenting with {name} LDA:")
    
    # Representation Strategy
    if vectorizer_type == 'BoW':
        vec = CountVectorizer(max_features=2000, max_df=0.95, min_df=2)
    else:
        vec = TfidfVectorizer(max_features=2000, max_df=0.95, min_df=2)
        
    X = vec.fit_transform(df['nlp_ready_text'])
    
    # LDA Model (Searching for 5 underlying topics)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    
    # Extract and print top words for each topic
    words = vec.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        top_words = [words[idx] for idx in topic.argsort()[-10:]]
        print(f"Topic {i}: {', '.join(top_words)}")
    
    return lda, vec

# Run both experiments
lda_bow, vec_bow = discover_topics('BoW', 'Bag of Words')
lda_tfidf, vec_tfidf = discover_topics('TF-IDF', 'TF-IDF')

# Strategy A: Bag of Words + LDA
print("Training BoW LDA:")
vec_bow = CountVectorizer(max_features=2000, max_df=0.95, min_df=2)
X_bow = vec_bow.fit_transform(df['nlp_ready_text'])

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X_bow)

# Strategy B: TF-IDF + LDA
print("Training TF-IDF LDA:")
vec_tfidf = TfidfVectorizer(max_features=2000, max_df=0.95, min_df=2)
X_tfidf = vec_tfidf.fit_transform(df['nlp_ready_text'])

lda_tfidf_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_tfidf_model.fit(X_tfidf)

print("Saving models: {'BoW LDA': lda_model, 'BoW Vectorizer': vec_bow, 'TF-IDF LDA': lda_tfidf_model, 'TF-IDF Vectorizer': vec_tfidf}")
joblib.dump(lda_model, '/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/lda_bow_model.pkl')
joblib.dump(vec_bow, '/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_bow.pkl')

joblib.dump(lda_tfidf_model, '/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/lda_tfidf_model.pkl')
joblib.dump(vec_tfidf, '/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_tfidf.pkl')