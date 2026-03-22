#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Visualization of LDA Topic Modelling results for both BoW and TF-IDF representations

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def plot_lda_comparison(csv_path, text_col):
    df = pd.read_csv(csv_path).dropna(subset=[text_col])
    
    experiments = [
        ('Bag of Words', CountVectorizer(max_features=2000)),
        ('TF-IDF', TfidfVectorizer(max_features=2000))
    ]
    
    for title, vectorizer in experiments:
        # Transform data
        X = vectorizer.fit_transform(df[text_col])
        
        # Fit LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        
        # Visualization
        words = vectorizer.get_feature_names_out()
        fig, axes = plt.subplots(1, 5, figsize=(22, 8))
        fig.suptitle(f"Top 10 Words: {title} Representation", fontsize=20, fontweight='bold')
        
        for i, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:]
            top_words = [words[idx] for idx in top_indices]
            weights = topic[top_indices]
            
            axes[i].barh(top_words, weights, color='teal' if 'TF' in title else 'coral')
            axes[i].set_title(f"Topic {i+1}", fontsize=14)
            axes[i].invert_yaxis()
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# run
plot_lda_comparison('nlp_processed.csv', 'nlp_ready_text')