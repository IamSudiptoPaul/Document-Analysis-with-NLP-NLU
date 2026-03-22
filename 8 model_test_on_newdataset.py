#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7059B Coursework 1

Testing the best trained model on a new dataset of social media posts and plotting

@author: Sudipto Paul (100538928)
@date:   12/03/2026

"""

import pandas as pd
import pickle
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_new_data(csv_path, text_col, label_col, model_path):
    # loading the original vectorizer
    df = pd.read_csv(csv_path)
    with open('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    X = vectorizer.transform(df[text_col].astype(str)).toarray()
    model = tf.keras.models.load_model(model_path)
    
    if len(model.input_shape) == 3:
        X = X.reshape(X.shape[0], X.shape[1], 1)

    # Predicting and plotting the matrix immediately
    y_pred = (model.predict(X) > 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(df[label_col], y_pred, display_labels=['Fake', 'Real'], cmap='Blues')
    
    plt.title(f"Test on: {csv_path}")
    plt.show()

# Run with specific file and column names, label 0 for Fake and 1 for Real, post_text for the text column
test_new_data('/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/test-dataset1.csv', 'post', 'class_label', '/Users/sudiptogoldfish/code files/7059B A_AI Lab/CW1/best_dl_model.keras')