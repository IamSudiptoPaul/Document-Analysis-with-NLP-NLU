#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Another approach for NLU: Advanced Semantic Topic Discovery using BGE-M3 (advanced NLU) and K-Means Clustering, 
saving the discovered clusters for later use.

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd
import numpy as np
import ssl
import joblib
import transformers.utils.import_utils as import_utils
import os
# This line tells TensorFlow to be quiet and stay in the background
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import transformers.utils.import_utils as import_utils
# Manually patching the missing function for FlagEmbedding compatibility
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False 

import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from sklearn.cluster import KMeans


# Patch for the ImportError
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False

from FlagEmbedding import BGEM3FlagModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Load a smaller sample for the demo to ensure it finishes quickly
df = pd.read_csv('nlp_processed.csv').sample(1000, random_state=42).reset_index(drop=True)

# use_fp16=True is crucial for Mac performance
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("Computing Semantic Vectors...")
embeddings = model.encode(df['nlp_ready_text'].tolist(), batch_size=8, show_progress_bar=True)['dense_vecs']

# Cluster these meanings into 5 topics
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df['semantic_topic'] = kmeans.fit_predict(embeddings)

print("\n--- BGE-M3 Semantic Topic Discovery ---")
for i in range(5):
    cluster_docs = df[df['semantic_topic'] == i]['nlp_ready_text']
    # Summarize the cluster using top keywords
    vec = TfidfVectorizer(max_features=8)
    vec.fit(cluster_docs)
    print(f"Topic {i}: {', '.join(vec.get_feature_names_out())}")

# Save the K-Means clusters discovered
joblib.dump(kmeans, 'semantic_clusters.pkl')
print("Semantic cluster model saved!")