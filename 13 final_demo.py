#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Testing the trained BoW LDA, TF-IDF LDA, and BGE-M3 Semantic Clustering models on new user input text

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import joblib
import os
import transformers.utils.import_utils as import_utils

# Environment Patches for Mac for BGE-M3
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from FlagEmbedding import BGEM3FlagModel

# Defined the Human-Readable Topic Names
TOPIC_NAMES = {
    0: "Climate & Energy",
    1: "General Discourse",
    2: "Public Health / COVID",
    3: "Economics & Labor",
    4: "Election / Politics"
}

def load_models():
    print("Loading Statistical Models BoW and TF-IDF:")
    lda_bow = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/lda_bow_model.pkl')
    vec_bow = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_bow.pkl')

    lda_tfidf = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/lda_tfidf_model.pkl')
    vec_tfidf = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer_tfidf.pkl')

    print("Loading Semantic Model (BGE-M3):")
    bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    kmeans_semantic = joblib.load('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/semantic_clusters.pkl')

    return lda_bow, vec_bow, lda_tfidf, vec_tfidf, bge_model, kmeans_semantic

def run_test():
    # Load all models
    l_bow, v_bow, l_tfidf, v_tfidf, bge, km = load_models()
    
    print("\n" + "="*40)
    print("READY:")
    print("="*40)
    
    while True:
        user_input = input("\nEnter text to analyze (or 'exit'): ")
        if user_input.lower() == 'exit': break
        
        print(f"\nResults for: '{user_input}'")
        print("-" * 30)

        # Test BoW
        bow_vec = v_bow.transform([user_input])
        bow_topic = l_bow.transform(bow_vec).argmax()
        print(f"BoW LDA Topic:      {bow_topic} ({TOPIC_NAMES.get(bow_topic)})")

        # Test TF-IDF
        tfidf_vec = v_tfidf.transform([user_input])
        tfidf_topic = l_tfidf.transform(tfidf_vec).argmax()
        print(f"TF-IDF LDA Topic:   {tfidf_topic} ({TOPIC_NAMES.get(tfidf_topic)})")

        # Test BGE-M3 Semantic Cluster
        semantic_vec = bge.encode([user_input])['dense_vecs']
        sem_topic = km.predict(semantic_vec)[0]
        print(f"BGE-M3 Topic:       {sem_topic} ({TOPIC_NAMES.get(sem_topic)})")

if __name__ == "__main__":
    run_test()


